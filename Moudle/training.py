import numpy as np
from global_parameters import MAX_SWAP, MAX_FRAGMENTS, GAMMA, BATCH_SIZE, EPOCHS, TIMES, FEATURES
from rewards import get_init_dist, evaluate_batch_mol, modify_fragment
import logging
import time

scores = 1. / TIMES
n_actions = MAX_FRAGMENTS * MAX_SWAP + 1


# Train actor and critic networks
def train(X, actor, critic, decodings):
    global frs
    hist = []
    dist_arr=[1.1,1.1,1.1]
    dist = np.array(dist_arr)
    # print("X:")
    # # print(X)
    # print("X_shape")
    # print(X.shape)

    # For every epoch
    for e in range(EPOCHS):

        # Select random starting "lead" molecules
        rand_n = np.random.randint(0, X.shape[0], BATCH_SIZE)
        rand_n[0:X.shape[0]] = np.arange(X.shape[0])
        batch_mol = X[rand_n].copy()
        r_tot = np.zeros(BATCH_SIZE)
        org_mols = batch_mol.copy()
        stopped = np.zeros(BATCH_SIZE) != 0

        # For all modification steps  对于所有修改步骤
        for t in range(TIMES):
            print(f"time:{t}")
            tm = (np.ones((BATCH_SIZE, 1)) * t) / TIMES

            probs = actor.predict([batch_mol, tm])
            print(probs)
            # 221*4个动作
            actions = np.zeros((BATCH_SIZE))
            rand_select = np.random.rand(BATCH_SIZE)
            old_batch = batch_mol.copy()
            rewards = np.zeros((BATCH_SIZE, 1))

            # Find probabilities for modification actions   查找修改action的概率
            for i in range(BATCH_SIZE):

                a = 0
                while True:
                    rand_select[i] -= probs[i, a]
                    if rand_select[i] < 0 or a + 1 == n_actions:
                        break
                    a += 1

                actions[i] = a

            # Initial critic value
            Vs = critic.predict([batch_mol, tm])

            # Select actions
            for i in range(BATCH_SIZE):

                a = int(actions[i])
                if stopped[i] or a == n_actions - 1:
                    stopped[i] = True
                    if t == 0:
                        rewards[i] += -1.
                    continue
                # 203
                t = int(a // 221)
                s = a % 221
                if batch_mol[i, t, 0] != s:
                    batch_mol[i, t, 0] = s
                else:
                    rewards[i] -= 0.1
            print(batch_mol)
            print(batch_mol.shape)
            # If final round  是最后一轮
            if t + 1 == TIMES:
                # frs = []
                print("asd")
                frs = evaluate_batch_mol(org_mols, batch_mol, e, decodings)
                if frs == -1:
                    n = 0
                    while frs == -1:
                        time.sleep(30)
                        frs = evaluate_batch_mol(org_mols, batch_mol, e, decodings)
                        n = n + 1
                        print('rerun ' + str(n) + ' times')
                for i in range(batch_mol.shape[0]):
                    # If molecule was modified
                    if not np.all(org_mols[i] == batch_mol[i]):
                        # frs= evaluate_batch_mol(org_mols,batch_mol, e, decodings)
                        rewards[i] += np.sum(frs[i] * dist)
                        if all(frs[i]):
                            rewards[i] *= 2
                    # else:
                    #    frs.append([False] * FEATURES)
                # Update distribution of rewards  更新奖励分配
                dist = 0.5 * dist + 0.5 * (1.0 / FEATURES * BATCH_SIZE / (1.0 + np.sum(frs, 0)))
            # Calculate TD-error
            target = rewards + GAMMA * critic.predict([batch_mol, tm + 1.0 / TIMES])
            td_error = target - Vs
            # Minimize TD-error 最小化TD-error
            critic.fit([old_batch, tm], target, verbose=0)
            target_actor = np.zeros_like(probs)
            for i in range(BATCH_SIZE):
                a = int(actions[i])
                loss = -np.log(probs[i, a]) * td_error[i]
                target_actor[i, a] = td_error[i]
            # Maximize expected reward.  最大化预期奖励
            actor.fit([old_batch, tm], target_actor, verbose=0)
            r_tot += rewards[:, 0]

        np.save("History/in-{}.npy".format(e), org_mols)
        np.save("History/out-{}.npy".format(e), batch_mol)
        # np.save("History/score-{}.npy".format(e), np.asarray(frs))

        hist.append([np.mean(r_tot)] + list(np.mean(frs, 0)) + [np.mean(np.sum(frs, 1) == 5)])
        print("Epoch {2} \t Mean score: {0:.3}\t\t Percentage in range: {1},  {3},td_error:{4}".format(
            np.mean(r_tot), [round(x, 2) for x in np.mean(frs, 0)], e,
            round(np.mean(np.sum(frs, 1) == 3)), np.mean(td_error, 0)
        ))

    return hist
