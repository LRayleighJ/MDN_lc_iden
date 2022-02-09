bm0_pre = np.append(bm0_pre, np.array(m0_batch)[np.argwhere(bspre_batch==1).T[0]] )
            bm0_act = np.append(bm0_act, np.array(m0_batch)[np.argwhere(label_batch==1).T[0]] )
            sm0_pre = np.append(sm0_pre, np.array(m0_batch)[np.argwhere(bspre_batch==0).T[0]] )
            sm0_act = np.append(sm0_act, np.array(m0_batch)[np.argwhere(label_batch==0).T[0]] )