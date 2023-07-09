from matplotlib.pyplot import cla, clf, Circle, figure, grid, legend, plot, savefig, show, subplot, title, xlabel, ylabel, fill_between
import numpy as np
from numpy import array, concatenate, dot, identity, linspace, ones, savetxt, size, sqrt, zeros
from numpy.random import uniform,seed
from numpy.random import permutation
from numpy import clip
from matplotlib import pyplot as plt

def plotTrainMetaData(alearn, atrue, aest, blearn, btrue, best, avar, bvar, ustd_list, residual_true_list, residual_pred_list,
                      residual_pred_lower_list, residual_pred_upper_list, residual_pred_compare_list, num_episodes, ebs, savedir, rnd_seed):
    for i in range(num_episodes-1):
        upper_a = alearn[i]*ustd_list[i] + 2*np.sqrt( avar[i] )*ustd_list[i]
        lower_a = alearn[i]*ustd_list[i] - 2*np.sqrt( avar[i] )*ustd_list[i]
    
        # plot a and b
        f = figure(figsize=(16, 4))
        subplot(241)
        plot(alearn[i]*ustd_list[i], label="Learned")
        fill_between(np.arange(alearn[i].size),lower_a, upper_a, color='blue', alpha=0.2)
        plot(atrue[i]*ustd_list[i], label="True")
        plot(aest[i]*ustd_list[i], label="Est")
        ylabel("a")
        title('Episode {}'.format(i+2))

        subplot(242)
        plot(avar[i]*ustd_list[i]**2)
        ylabel("a_var") 

        upper_b = blearn[i] + 2*np.sqrt(bvar[i])
        lower_b = blearn[i] - 2*np.sqrt(bvar[i])
      
        subplot(243)  
        plot(blearn[i].squeeze(-1), label="Learned")
        fill_between(np.arange(blearn[i].size), lower_b.squeeze(-1), upper_b.squeeze(-1), color='blue', alpha=0.2)
        plot(btrue[i], label="True")
        plot(best[i], label="Est")
        ylabel("b")  
        title('Episode {}'.format(i+2))  

        subplot(244)
        plot(bvar[i])
        ylabel("b_var") 

        subplot(245)  
        plot(alearn[i]*ustd_list[i]-aest[i]*ustd_list[i], label="Learned")
        fill_between(np.arange(alearn[i].size),lower_a-aest[i]*ustd_list[i], upper_a-aest[i]*ustd_list[i], color='blue', alpha=0.2)
        plot(atrue[i]*ustd_list[i]-aest[i]*ustd_list[i], label="True")
        ylabel("a_residual")

        subplot(246)  
        plot(blearn[i].squeeze(-1)-best[i], label="Learned")
        fill_between(np.arange(blearn[i].size),lower_b.squeeze(-1)-best[i], upper_b.squeeze(-1)-best[i], color='blue', alpha=0.2)
        plot(btrue[i]-best[i], label="True")
        ylabel("b_residual")  

        subplot(247)  
        # plot only the last episode of residuals
        plot(residual_pred_list[i][-ebs:], label="Learned")
        fill_between(np.arange(residual_pred_list[i][-ebs:].size),residual_pred_lower_list[i][-ebs:], residual_pred_upper_list[i][-ebs:], color='blue', alpha=0.2)
        plot(residual_true_list[i][-ebs:], label="True")
        plot(residual_pred_compare_list[i][-ebs:])
        ylabel("residual")  
        legend(["Learned","True","Predict compute"])

        f.savefig(savedir+"/ep{}_learned_ab_seed{}.pdf".format(str(i+2), str(rnd_seed)), bbox_inches='tight')
        f.savefig(savedir+"/ep{}_learned_ab_seed{}.png".format(str(i+2), str(rnd_seed)), bbox_inches='tight')

def plotQuadStates(ts_qp, ts_post_qp, xs_qp_trueest, xs_qp_truetrue, xs_post_qp, us_qp_trueest, us_qp_truetrue, us_post_qp, hs_qp_trueest, hs_qp_truetrue, hs_post_qp, hdots_post_qp, hdots_true_post_qp, hdots_learned_post_qp , drifts_post_qp, drifts_true_post_qp, drifts_learned_post_qp, acts_post_qp, acts_true_post_qp, acts_learned_post_qp, savename):
    
    f = figure(figsize=(16, 16))
    subplot(521)
    cla()
    plot(ts_qp, xs_qp_trueest[:, 0], 'g', linewidth=3, label='TrueEst')
    plot(ts_qp, xs_qp_truetrue[:, 0], 'r', linewidth=3, label='TrueTrue')
    plot(ts_post_qp, xs_post_qp[:, 0], 'b', linewidth=3, label='TrueLearned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$x (m)$', fontsize=16)
    legend(fontsize =16)

    subplot(522)
    plot(ts_qp, xs_qp_trueest[:, 1], 'g', linewidth=3, label='TrueEst')
    plot(ts_qp, xs_qp_truetrue[:, 1], 'r', linewidth=3, label='TrueTrue')
    plot(ts_post_qp, xs_post_qp[:, 1], 'b', linewidth=3, label='TrueLearned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$y (m)$', fontsize=16)
    legend(fontsize =16)

    subplot(523)
    plot(ts_qp, hs_qp_trueest, 'g', linewidth=3, label = 'TrueEst')
    plot(ts_qp, hs_qp_truetrue, 'r', linewidth=3, label = 'TrueTrue')
    plot(ts_post_qp, hs_post_qp, 'b', linewidth=3, label = 'TrueLearned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$h$', fontsize=16)
    legend(fontsize =16)

    subplot(524)
    plot(ts_qp[:-1], us_qp_trueest[:, 0], 'g', linewidth=3, label = 'TrueEst')
    plot(ts_qp[:-1], us_qp_truetrue[:, 0], 'r', linewidth=3, label = 'TrueTrue')
    plot(ts_post_qp[:-1], us_post_qp[:, 0], 'b', linewidth=3, label='TrueLearned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$\\tau_1$', fontsize=16)
    legend(fontsize=16)
    
    subplot(525)
    plot(ts_qp[:-1], us_qp_trueest[:, 1], 'g', linewidth=3, label = 'TrueEst')
    plot(ts_qp[:-1], us_qp_truetrue[:, 1], 'r', linewidth=3, label = 'TrueTrue')
    plot(ts_post_qp[:-1], us_post_qp[:, 1], 'b', linewidth=3, label='TrueLearned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$\\tau_2$', fontsize=16)
    legend(fontsize=16)

    subplot(526)
    plot(ts_post_qp[:-1], hdots_post_qp, 'g', linewidth=3, label='Est')
    plot(ts_post_qp[:-1], hdots_true_post_qp , 'r', linewidth=3, label='True')
    # plot(ts_post_qp[1:-1], hdots_post_num, 'b', linewidth=3, label='Numerical Estimate')
    plot(ts_post_qp[:-1], hdots_learned_post_qp, 'b', linewidth=3, label = 'Learned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$\\dot{h}$', fontsize=16)
    legend(fontsize=16)

    subplot(527)
    plot(ts_post_qp, drifts_post_qp, 'g', linewidth=3, label='Est')
    plot(ts_post_qp, drifts_true_post_qp , 'r', linewidth=3, label='True')
    plot(ts_post_qp, drifts_learned_post_qp, 'b', linewidth=3, label = 'Learned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$L_fh$', fontsize=16)
    legend(fontsize=16)

    subplot(528)
    plot(ts_post_qp, acts_post_qp[:,0], 'g', linewidth=3, label='Est')
    plot(ts_post_qp, acts_true_post_qp[:,0] , 'r', linewidth=3, label='True')
    plot(ts_post_qp, acts_learned_post_qp[:,0], 'b', linewidth=3, label = 'Learned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$L_gh$', fontsize=16)
    legend(fontsize=16)
    
    subplot(529)
    plot(ts_post_qp, acts_post_qp[:,1], 'g', linewidth=3, label='Est')
    plot(ts_post_qp, acts_true_post_qp[:,1] , 'r', linewidth=3, label='True')
    plot(ts_post_qp, acts_learned_post_qp[:,1], 'b', linewidth=3, label = 'Learned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$L_gh$', fontsize=16)
    legend(fontsize=16)
    
    f.savefig(savename, bbox_inches='tight')

def plotTestStates(ts_qp, ts_post_qp, xs_qp_trueest, xs_qp_truetrue, xs_post_qp, us_qp_trueest, us_qp_truetrue, us_post_qp, hs_qp_trueest, hs_qp_truetrue, hs_post_qp, hdots_post_qp, hdots_true_post_qp, hdots_learned_post_qp , drifts_post_qp, drifts_true_post_qp, drifts_learned_post_qp, acts_post_qp, acts_true_post_qp, acts_learned_post_qp, theta_bound_u, theta_bound_l, savename):
    
    f = figure(figsize=(16, 16))
    subplot(521)
    cla()
    plot(ts_qp, xs_qp_trueest[:, 0], 'g', linewidth=3, label='TrueEst')
    plot(ts_qp, xs_qp_truetrue[:, 0], 'r', linewidth=3, label='TrueTrue')
    plot(ts_post_qp, xs_post_qp[:, 0], 'b', linewidth=3, label='TrueLearned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$x (m)$', fontsize=16)
    legend(fontsize =16)

    subplot(522)
    plot(ts_qp, xs_qp_trueest[:, 1], 'g', linewidth=3, label='TrueEst')
    plot(ts_qp, xs_qp_truetrue[:, 1], 'r', linewidth=3, label='TrueTrue')
    plot(ts_post_qp, xs_post_qp[:, 1], 'b', linewidth=3, label='TrueLearned')
    plot(ts_post_qp, theta_bound_u, '--k', linewidth = 3)
    plot(ts_post_qp, theta_bound_l, '--k', linewidth = 3)
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$\\theta (rad)$', fontsize=16)
    legend(fontsize =16)

    subplot(523)
    cla()
    plot(ts_qp, xs_qp_trueest[:, 2], 'g', linewidth=3, label='TrueEst')
    plot(ts_qp, xs_qp_truetrue[:, 2], 'r', linewidth=3, label='TrueTrue')
    plot(ts_post_qp, xs_post_qp[:, 2], 'b', linewidth=3, label='TrueLearned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$\\dot{x} (m/s)$', fontsize=16)
    legend(fontsize =16)

    subplot(524)
    plot(ts_qp, xs_qp_trueest[:, 3], 'g', linewidth=3, label='TrueEst')
    plot(ts_qp, xs_qp_truetrue[:, 3], 'r', linewidth=3, label='TrueTrue')
    plot(ts_post_qp, xs_post_qp[:, 3], 'b', linewidth=3, label='TrueLearned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$\\dot{\\theta} (rad/s)$', fontsize=16)
    legend(fontsize =16)

    subplot(525)
    plot(ts_qp, hs_qp_trueest, 'g', linewidth=3, label = 'TrueEst')
    plot(ts_qp, hs_qp_truetrue, 'r', linewidth=3, label = 'TrueTrue')
    plot(ts_post_qp, hs_post_qp, 'b', linewidth=3, label = 'TrueLearned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$h$', fontsize=16)
    legend(fontsize =16)

    subplot(526)
    plot(ts_qp[:-1], us_qp_trueest[:, 0], 'g', linewidth=3, label = 'TrueEst')
    plot(ts_qp[:-1], us_qp_truetrue[:, 0], 'r', linewidth=3, label = 'TrueTrue')
    plot(ts_post_qp[:-1], us_post_qp[:, 0], 'b', linewidth=3, label='TrueLearned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$\\tau_1$', fontsize=16)
    legend(fontsize=16)

    subplot(527)
    plot(ts_post_qp[:-1], hdots_post_qp, 'g', linewidth=3, label='Est')
    plot(ts_post_qp[:-1], hdots_true_post_qp , 'r', linewidth=3, label='True')
    # plot(ts_post_qp[1:-1], hdots_post_num, 'b', linewidth=3, label='Numerical Estimate')
    plot(ts_post_qp[:-1], hdots_learned_post_qp, 'b', linewidth=3, label = 'Learned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$\\dot{h}$', fontsize=16)
    legend(fontsize=16)

    subplot(528)
    plot(ts_post_qp, drifts_post_qp, 'g', linewidth=3, label='Est')
    plot(ts_post_qp, drifts_true_post_qp , 'r', linewidth=3, label='True')
    plot(ts_post_qp, drifts_learned_post_qp, 'b', linewidth=3, label = 'Learned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$L_fh$', fontsize=16)
    legend(fontsize=16)

    subplot(529)
    plot(ts_post_qp, acts_post_qp, 'g', linewidth=3, label='Est')
    plot(ts_post_qp, acts_true_post_qp , 'r', linewidth=3, label='True')
    plot(ts_post_qp, acts_learned_post_qp, 'b', linewidth=3, label = 'Learned')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$L_gh$', fontsize=16)
    legend(fontsize=16)
    
    f.savefig(savename, bbox_inches='tight')

def plotLearnedCBF(ts_qp, hs_qp_trueest, hs_all, ts_post_qp, hs_post_qp, ebs, num_episodes, savename):
    f = figure(figsize=(10, 8))
    # # Initial Result
    plot(ts_qp, hs_qp_trueest, 'g', linewidth=3, label='Model Based')

    # # Intermediate Results
    for j in range(num_episodes):
        plot(ts_qp, hs_all[j*ebs:(j+1)*ebs], 'r', linewidth = 3, alpha = 0.4*0.5**(num_episodes-j))

    # # Learned Results
    plot(ts_post_qp, hs_post_qp, 'b', linewidth=3, label = 'Learned')

    # grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$h$', fontsize=16)
    title('Learned Controller', fontsize = 24)
    legend(fontsize = 16)
    show()
    f.savefig(savename, bbox_inches='tight')

def plotPhasePlane(theta_h0_vals, theta_dot_h0_vals, xs_qp_trueest, state_data, xs_post_qp, ebs, num_episodes, savename):
    # LEARNED CONTROLLER
    f = figure(figsize=(10, 8))
    # Safe Set
    plot(theta_h0_vals, theta_dot_h0_vals, 'k', linewidth=3, label='$\partial S$')
    plot(theta_h0_vals, -theta_dot_h0_vals, 'k', linewidth=3)
    # Initial Result
    plot(xs_qp_trueest[:, 1], xs_qp_trueest[:, 3], 'g', linewidth=3, label='Model Based')
    # Intermediate Results
    for j in range(num_episodes):
        plot(state_data[0][j*ebs:(j+1)*ebs,1], state_data[0][j*ebs:(j+1)*ebs,3], 'r', linewidth = 3, alpha = 0.4*0.5**(num_episodes-j))
    # Final Result
    plot(xs_post_qp[:, 1], xs_post_qp[:, 3], 'b', linewidth=3, label='Learned')
    grid()
    xlabel('$\\theta (rad)$', fontsize=16)
    ylabel('$\\dot{\\theta} (rad/s)$', fontsize=16)
    title('Learned Controller', fontsize = 24)
    legend(fontsize = 16)
    show()
    f.savefig(savename, bbox_inches='tight')

def plotQuadTrajectory(state_data, num_episodes, xs_post_qp, xs_qp_trueest, xs_qp_truetrue, x_e, y_e, rad, savename, title_label='ProBF-GP'):
    ebs = int(len(state_data[0])/num_episodes)
    # Intermediate Results
    f = figure(figsize=(10, 8))
    plot(state_data[0][0*ebs:1*ebs,0], state_data[0][0*ebs:1*ebs,1], 'r', linewidth = 2, alpha = 0.3, label="Episode 1")
    plot(state_data[0][2*ebs:3*ebs,0], state_data[0][2*ebs:3*ebs,1], 'r', linewidth = 2, alpha = 0.3, label="Episode 3")
    #plot(state_data[0][1*ebs:2*ebs,0], state_data[0][1*ebs:2*ebs,1], 'r', linewidth = 2, alpha = 0.5, label="Episode 2")
    plot(state_data[0][6*ebs:7*ebs,0], state_data[0][6*ebs:7*ebs,1], 'r', linewidth = 2, alpha = 0.7, label="Episode 7")
    #plot(state_data[0][7*ebs:8*ebs,0], state_data[0][7*ebs:8*ebs,1], 'r', linewidth = 2, alpha = 0.8, label="Episode 8")
    plot(xs_post_qp[:, 0], xs_post_qp[:, 1], 'b', linewidth=2, label=title_label)
    plot(xs_qp_trueest[:, 0], xs_qp_trueest[:, 1], 'g', label='Nominal Model')
    #plot(xs_qp_truetrue[:, 0], xs_qp_truetrue[:, 1], 'c', label='True-True-QP')
    circle = Circle((x_e,y_e),0.9*np.sqrt(rad),color="y")
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_patch(circle)

    grid()
    legend(fontsize=18)
    xlabel('$x$', fontsize=18)
    ylabel('$y$', fontsize=18)
    plt.xlim([-3,3])
    plt.ylim([-1,2.5])
    f.savefig(savename, bbox_inches='tight') 
    
def plotTrainStates(input_data_list, ebs_res, num_episodes, savedir, rnd_seed):
    for i in range(num_episodes-1):
        # plot states and controls
        f = figure(figsize=(16, 4))  
        subplot(241)
        plot(input_data_list[i][-ebs_res:,1])
        ylabel("state(0)") 

        subplot(242)
        plot(input_data_list[i][-ebs_res:,2])
        ylabel("state(1)") 

        subplot(243)
        plot(input_data_list[i][-ebs_res:,3])
        ylabel("state(2)") 

        subplot(244)
        plot(input_data_list[i][-ebs_res:,4])
        ylabel("state(3)") 

        subplot(245)
        plot(input_data_list[i][-ebs_res:,6])
        ylabel("dh/dx(1)") 

        subplot(246)
        plot(input_data_list[i][-ebs_res:,8])
        ylabel("dh/dx(3)") 

        subplot(247)
        plot(input_data_list[i][-ebs_res:,0])
        ylabel("u") 
      
        f.savefig(savedir+"ep{}_state_seed{}.pdf".format(str(i+2), str(rnd_seed)), bbox_inches='tight')
