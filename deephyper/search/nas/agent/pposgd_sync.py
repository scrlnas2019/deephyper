import multiprocessing as mp

import time
from collections import deque
from pprint import pformat

import numpy as np
import tensorflow as tf
from mpi4py import MPI

import deephyper.search.nas.utils.common.tf_util as U
from deephyper.search import util
from deephyper.search.nas.agent.utils import (reward_for_final_timestep,
                                              traj_segment_generator_ph)
from deephyper.search.nas.utils import logger
from deephyper.search.nas.utils._logging import JsonMessage as jm
from deephyper.search.nas.utils.common import (Dataset, explained_variance,
                                               fmt_row, zipsame)
from deephyper.search.nas.utils.common.mpi_adam import MpiAdam
from deephyper.search.nas.utils.common.mpi_moments import mpi_moments

dh_logger = util.conf_logger('deephyper.search.nas.agent.pposgd_sync')

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        reward_rule=reward_for_final_timestep
        ):

    rank = MPI.COMM_WORLD.Get_rank()

    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    input_c_vf = U.get_placeholder_cached(name="c_vf")
    input_h_vf = U.get_placeholder_cached(name="h_vf")
    input_c_pol = U.get_placeholder_cached(name="c_pol")
    input_h_pol = U.get_placeholder_cached(name="h_pol")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function(
        [ob, ac, atarg, ret, lrmult, input_c_vf, input_h_vf, input_c_pol, input_h_pol],
        losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult, input_c_vf, input_h_vf, input_c_pol, input_h_pol], losses)

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator_ph(pi, env, timesteps_per_actorbatch,
        stochastic=True, reward_affect_func=reward_rule)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert max_timesteps > 0, f"The number of timesteps should be > 0 but is {max_timesteps}"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        dh_logger.info(jm(type='seg', rank=rank, **seg))
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        c_vf = np.squeeze(np.array([c for c, _ in seg["hs_vf"]]))
        h_vf = np.squeeze(np.array([h for _, h in seg["hs_vf"]]))
        c_pol = np.squeeze(np.array([c for c, _ in seg["hs_pol"]]))
        h_pol = np.squeeze(np.array([h for _, h in seg["hs_pol"]]))
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, c_vf=c_vf, h_vf=h_vf, c_pol=c_pol, h_pol=h_pol), shuffle=not pi.recurrent)
        # optim_batchsize = optim_batchsize or ob.shape[0]
        optim_batchsize = ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            gradients = []
            for batch in d.iterate_once(optim_batchsize):
                for i in range(len(batch["ob"])):
                    *newlosses, g = lossandgrad(
                        batch["ob"][i:i+1],
                        batch["ac"][i:i+1],
                        batch["atarg"][i:i+1],
                        batch["vtarg"][i:i+1],
                        cur_lrmult,
                        batch["c_vf"][i:i+1],
                        batch["h_vf"][i:i+1],
                        batch["c_pol"][i:i+1],
                        batch["h_pol"][i:i+1])
                    losses.append(newlosses)
                    gradients.append(g)
            g = np.array(gradients).sum(0)
            adam.update(g, optim_stepsize * cur_lrmult)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            for i in range(len(batch["ob"])):
                newlosses = compute_losses(
                    batch["ob"][i:i+1],
                    batch["ac"][i:i+1],
                    batch["atarg"][i:i+1],
                    batch["vtarg"][i:i+1],
                    cur_lrmult,
                    batch["c_vf"][i:i+1],
                    batch["h_vf"][i:i+1],
                    batch["c_pol"][i:i+1],
                    batch["h_pol"][i:i+1])
                losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
