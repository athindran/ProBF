from matplotlib.pyplot import cla, Circle, figure, grid, legend, plot, show, subplot, title, xlabel, ylabel, fill_between, close
import numpy as np
from matplotlib import pyplot as plt

import imageio
import os

def make_animation(xs_array, x_d, obstacle_position, obstacle_rad2, fig_folder='./animation'):  
  """
    Animate array of positions:
    Inputs:
        xs_array: Array of time varying quadrotor position
        x_d: Desired Quadrotor location
        obstacle_position: Position of obstacle
        obstacle_rad2: Radius^{2} of obstacle
  """
  if not os.path.isdir(fig_folder):
      os.mkdir(fig_folder)

  frame_skip = 20
  for i in range(0, xs_array.shape[0], frame_skip):
    quadrotor_position = xs_array[i][0:2]
    quadrotor_orientation = xs_array[i][2] 
    length = 0.5

    quadrotor_left_x = quadrotor_position[0] - length*np.cos(quadrotor_orientation)
    quadrotor_left_y = quadrotor_position[1] + length*np.sin(quadrotor_orientation)
    quadrotor_right_x = quadrotor_position[0] + length*np.cos(quadrotor_orientation)
    quadrotor_right_y = quadrotor_position[1] - length*np.sin(quadrotor_orientation)
    f = plt.figure(figsize=(6, 4))
    plt.plot(x_d[0, :], x_d[1, :], 'k*', label='Desired')
    plt.plot([quadrotor_left_x, quadrotor_right_x], [quadrotor_left_y, quadrotor_right_y], 'x-')
    circle = Circle((obstacle_position[0], obstacle_position[1]), np.sqrt(obstacle_rad2) - 1.0, color="y")
    ax = f.gca()
    ax.add_patch(circle)
    ax.set_xticks([-2.0, obstacle_position[0], x_d[0, 0], 10.0])
    ax.set_yticks([-2.0, obstacle_position[1], x_d[1, 0], 11.0])
    ax.set_ylabel('Y position')
    ax.set_xlabel('X position')
    ax.set_xlim([-2.0, 10.0])
    ax.set_ylim([-2.0, 11.0])
    ax.legend()
    plt.savefig(fig_folder + str(i) + '.png')
    plt.close()
  
  gif_path = fig_folder + 'rollout.gif'
  with imageio.get_writer(gif_path, mode='I') as writer:
    for i in range(0, xs_array.shape[0], frame_skip):
        filename = os.path.join(fig_folder, str(i) + ".png")
        image = imageio.imread(filename)
        writer.append_data(image)

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

        close()

def plotQuadStates(ts_qp, ts_post_qp, xs_qp_trueest, xs_qp_truetrue, xs_post_qp, us_qp_trueest, us_qp_truetrue,
                    us_post_qp, hs_qp_trueest, hs_qp_truetrue, hs_post_qp, hdots_post_qp, hdots_true_post_qp, hdots_learned_post_qp , 
                    drifts_post_qp, drifts_true_post_qp, drifts_learned_post_qp, acts_post_qp, acts_true_post_qp, acts_learned_post_qp, savename):
    
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

    close()


def plotQuadStatesv2(axes2, ts_qp, xs_qp, us_qp, hs_qp, hdots_qp, label='TrueEst', clr='r'):
    from core.util import differentiate
    lw = 2
    fs = 12

    hdots_num_qp = differentiate(hs_qp, ts_qp)
    axes2[0, 0].plot(ts_qp, xs_qp[:, 0], color=clr, linewidth=lw, label = label)
    axes2[0, 0].grid(visible=True)
    axes2[0 ,0].set_xlabel('Time (sec)', fontsize=fs)
    axes2[0 ,0].set_ylabel('$x (m)$', fontsize=fs)
    axes2[0 ,0].legend(fontsize =fs)

    axes2[0, 1].plot(ts_qp, xs_qp[:, 1], color=clr, linewidth=lw, label = label)
    axes2[0, 1].grid(visible=True)
    axes2[0, 1].set_xlabel('Time (sec)', fontsize=fs)
    axes2[0, 1].set_ylabel('$y (m)$', fontsize=fs)
    axes2[0, 1].legend(fontsize=fs)

    axes2[0, 2].plot(ts_qp, hs_qp, color=clr, linewidth=lw, label = label)
    axes2[0, 2].grid(visible=True)
    axes2[0, 2].set_xlabel('Time (sec)', fontsize=fs)
    axes2[0, 2].set_ylabel('$h$', fontsize=fs)
    axes2[0, 2].set_ylim([-1, 10])
    axes2[0, 2].legend(fontsize=fs)

    axes2[1, 0].plot(ts_qp[:-1], us_qp[:, 0], color=clr, linewidth=lw, label = label)
    axes2[1, 0].grid(visible=True)
    axes2[1, 0].set_xlabel('Time (sec)', fontsize=fs)
    axes2[1, 0].set_ylabel('$\\tau_1$', fontsize=fs)
    axes2[1, 0].legend(fontsize=16)
    
    axes2[1, 1].plot(ts_qp[:-1], us_qp[:, 1], color=clr, linewidth=lw, label = label)
    axes2[1, 1].grid(visible=True)
    axes2[1, 1].set_xlabel('Time (sec)', fontsize=fs)
    axes2[1, 1].set_ylabel('$\\tau_2$', fontsize=fs)
    axes2[1, 1].legend(fontsize=fs)

    axes2[1, 2].plot(ts_qp[:-1], hdots_qp, color=clr, linewidth=lw, label=label)
    axes2[1, 2].plot(ts_qp[1:-1], hdots_num_qp, color=clr, linestyle='dashed', linewidth=lw, label=label)
    axes2[1, 2].grid(visible=True)
    axes2[1, 2].set_xlabel('Time (sec)', fontsize=fs)
    axes2[1, 2].set_ylabel('$\\dot{h}$', fontsize=fs)
    axes2[1, 2].legend(fontsize=fs)

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

    close()

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
    f.savefig(savename, bbox_inches='tight')

    close()

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
    f.savefig(savename, bbox_inches='tight')

    close()

def plotPredictions(safety_learned, data_episode, savename, device='cpu'):
    """
    Plots the comparison of residual predictions and actual residuals.

    Inputs:
        safety_learned: Learned safety filter
        data_episode: Trajectory data from each episode
    """
    import torch
    import gpytorch
    
    residual_model = safety_learned.residual_model
    likelihood = safety_learned.likelihood
    
    (drift_inputs, _, us, residuals, _) = data_episode
    npoints = drift_inputs.shape[0]
    test_inputs = (torch.from_numpy( drift_inputs ) - torch.reshape(safety_learned.preprocess_mean, (-1, 8)).repeat(npoints, 1) )
    test_inputs = torch.divide(test_inputs, torch.reshape(safety_learned.preprocess_std, (-1, 8)).repeat(npoints, 1) )
    test_inputs = torch.cat((torch.from_numpy(us/safety_learned.us_scale), test_inputs), axis=1)
    test_inputs = test_inputs.float().to(device)

    residual_model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_computations(solves=False):
        residual_pred = likelihood(safety_learned.residual_model(test_inputs))
    lower, upper = residual_pred.confidence_region()
    lower = lower.cpu()
    upper = upper.cpu()
    mean = residual_pred.mean.detach().cpu().numpy()
    var = residual_pred.variance.detach().cpu().numpy()
    f = figure()
    plot(mean*safety_learned.residual_std + safety_learned.residual_mean)
    fill_between(np.arange(mean.size), lower.detach().numpy(), upper.detach().numpy(), color='blue', alpha=0.2)
    plot(residuals)
    xlabel('Time')
    ylabel('CBF residuals')
    legend(["Prediction", "Actual"])
    f.savefig(savename, bbox_inches='tight')

    close()

def plotQuadTrajectory(state_data, num_episodes, xs_post_qp, xs_qp_trueest, xs_qp_truetrue, 
                       obstacle_position, rad_square, x_d,
                       savename, title_label='ProBF-GP'):
    ebs = int(len(state_data[0])/num_episodes)
    # Intermediate Results
    f = figure(figsize=(10, 8))
    plot(state_data[0][0*ebs:1*ebs,0], state_data[0][0*ebs:1*ebs,1], 'r', linewidth = 2, alpha = 0.3, label="Episode 1")
    plot(state_data[0][2*ebs:3*ebs,0], state_data[0][2*ebs:3*ebs,1], 'r', linewidth = 2, alpha = 0.3, label="Episode 3")
    #plot(state_data[0][1*ebs:2*ebs,0], state_data[0][1*ebs:2*ebs,1], 'r', linewidth = 2, alpha = 0.5, label="Episode 2")
    #plot(state_data[0][6*ebs:7*ebs,0], state_data[0][6*ebs:7*ebs,1], 'r', linewidth = 2, alpha = 0.7, label="Episode 7")
    #plot(state_data[0][7*ebs:8*ebs,0], state_data[0][7*ebs:8*ebs,1], 'r', linewidth = 2, alpha = 0.8, label="Episode 8")
    plot(xs_post_qp[:, 0], xs_post_qp[:, 1], 'b', linewidth=2, label=title_label)
    plot(xs_qp_trueest[:, 0], xs_qp_trueest[:, 1], 'g', label='Nominal Model')
    plot(xs_qp_truetrue[:, 0], xs_qp_truetrue[:, 1], 'c', label='True model')
    circle = Circle((obstacle_position[0], obstacle_position[1]), np.sqrt(rad_square),color="y")
    ax = f.gca()
    ax.add_patch(circle)
    ax.plot(x_d[0, :], x_d[1, :], 'k*', label='Desired')
    ax.set_xticks([-2, obstacle_position[0], x_d[0, 0], 13])
    ax.set_yticks([-2, obstacle_position[1], x_d[1, 0], 13])
    ax.set_ylabel('Y position')
    ax.set_xlabel('X position')
    ax.set_xlim([-2, 13])
    ax.set_ylim([-2, 13])
    ax.legend()
    f.savefig(savename, bbox_inches='tight') 

    close()
    
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

        close()