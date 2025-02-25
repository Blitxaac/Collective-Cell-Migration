# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 09:34:16 2025

1. The first section plots the effects of parameters on avg velocity and polarity (split into 85 regions along the x-axis).
Two frames are plotted, the midpoint frame and the final frame.

2. The second section plots the time evolution of avg polarity and x-velocity for each parameter set

3.
"""
import re
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

def file_create(script_direc):
    save_directory1 = os.path.join(script_direc, "vel vs pos")

    if not os.path.exists(save_directory1):
        os.makedirs(save_directory1)
        
    save_directory2 = os.path.join(script_direc, "pol vs pos")

    if not os.path.exists(save_directory2):
        os.makedirs(save_directory2)
    
    frame_path1 = os.path.normpath(save_directory1)
    frame_path2 = os.path.normpath(save_directory2)
    
    return frame_path1,frame_path2

def average_in_regions(data, num_regions):
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    bins = np.linspace(x_min, x_max, num_regions + 1)  # Create bin edges
    indices = np.digitize(data[:, 0], bins)  # Assign x values to bins

    averages = []
    bin_centers = []

    for i in range(1, num_regions + 1):
        mask = indices == i  # Select data in the region
        if np.any(mask):  # Avoid empty regions
            avg_y = np.mean(data[mask, 1])  
            bin_center = (bins[i - 1] + bins[i]) / 2  # Compute bin center
        else:
            avg_y = np.nan
            bin_center = (bins[i - 1] + bins[i]) / 2  # Still keep the x position

        averages.append(avg_y)
        bin_centers.append(bin_center)

    return np.array(bin_centers), np.array(averages)

def velplot(vel_path,pos_path,vel_direc,num_regions=20):
    plt.figure(figsize=(12,6))

    # First subplot for frame 1
    plt.subplot(1, 2, 1)  # (rows, columns, index)
    with h5py.File(vel_path, "r") as f1, h5py.File(pos_path, "r") as f2:
        
        dataset_names = list(f1.keys())
        # Sort the dataset names numerically
        dataset_names_sorted = sorted(dataset_names, key=lambda x: int(re.search(r"\d+", x).group()))
        leng = len(dataset_names_sorted)
        # Select the middle frame and the final frame
        frame2 = dataset_names_sorted[leng-1]
        frame1 = dataset_names_sorted[round((leng-1)/2)]
        
        ve1 = f1[frame1][:]  # Read data for frame1
        veldata1 = np.sqrt(ve1[:, 0:1]**2 + ve1[:, 1:2]**2).reshape(-1, 1)
        posdata1 = f2[frame1][:][:, 0:1]
        data1 = np.hstack((posdata1, veldata1))
        bin_centers1, avg_values1 = average_in_regions(data1, num_regions)

        # Plot frame 1 data in the first subplot
        #plt.scatter(bin_centers1, avg_values1, color='red', label='Region averages', s=1)
        plt.plot(bin_centers1,avg_values1,marker='o', linestyle='-',label='Region averages',color='red')
        plt.xlabel('Position (mm)')
        plt.ylabel('Velocity')
        plt.title(f'Frame {frame1}')
        plt.legend()

    # Second subplot for frame 2
        plt.subplot(1, 2, 2)  # (rows, columns, index)
        ve2 = f1[frame2][:]  # Read data for frame2
        veldata2 = np.sqrt(ve2[:, 0:1]**2 + ve2[:, 1:2]**2).reshape(-1, 1)
        posdata2 = f2[frame2][:][:, 0:1]
        data2 = np.hstack((posdata2, veldata2))
        bin_centers2, avg_values2 = average_in_regions(data2, num_regions)

        # Plot frame 2 data in the second subplot
        #plt.scatter(bin_centers2, avg_values2, color='blue', label='Region averages', s=1)
        plt.plot(bin_centers2,avg_values2,marker='o', linestyle='-',label='Region averages')
        plt.xlabel('Position (mm)')
        plt.ylabel('Velocity')
        plt.title(f'Frame {frame2}')
        plt.legend()

        vel_filename = os.path.basename(vel_path)
        frame_path = os.path.join(vel_direc, f'{vel_filename}.png')
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(frame_path)  # Save image with both subplots
        plt.close()

def polplot(pol_path,pos_path,pol_direc,num_regions=20):
    plt.figure(figsize=(12,6))

    # First subplot for frame 1
    plt.subplot(1, 2, 1)  # (rows, columns, index)
    with h5py.File(pol_path, "r") as f1, h5py.File(pos_path, "r") as f2:
        dataset_names = list(f1.keys())
        # Sort the dataset names numerically
        dataset_names_sorted = sorted(dataset_names, key=lambda x: int(re.search(r"\d+", x).group()))
        leng = len(dataset_names_sorted)
        # Select the middle frame and the final frame
        frame2 = dataset_names_sorted[leng-1]
        frame1 = dataset_names_sorted[round((leng-1)/2)]
        
        po1 = f1[frame1][:]  # Read data for frame1
        poldata1 = np.sqrt(po1[:, 0:1]**2 + po1[:, 1:2]**2).reshape(-1, 1)
        posdata1 = f2[frame1][:][:, 0:1]
        data1 = np.hstack((posdata1, poldata1))
        bin_centers1, avg_values1 = average_in_regions(data1, num_regions)

        # Plot frame 1 data in the first subplot
        #plt.scatter(bin_centers1, avg_values1, color='red', label='Region averages', s=1)
        plt.plot(bin_centers1,avg_values1,marker='o', linestyle='-',label='Region averages',color='red')
        plt.xlabel('Position (mm)')
        plt.ylabel('Polarity')
        plt.title(f'Frame {frame1}')
        plt.legend()

    # Second subplot for frame 2
        plt.subplot(1, 2, 2)  # (rows, columns, index)
        po2 = f1[frame2][:]  # Read data for frame2
        poldata2 = np.sqrt(po2[:, 0:1]**2 + po2[:, 1:2]**2).reshape(-1, 1)
        posdata2 = f2[frame2][:][:, 0:1]
        data2 = np.hstack((posdata2, poldata2))
        bin_centers2, avg_values2 = average_in_regions(data2, num_regions)

        # Plot frame 2 data in the second subplot
        #plt.scatter(bin_centers2, avg_values2, color='blue', label='Region averages', s=1)
        plt.plot(bin_centers2,avg_values2,marker='o', linestyle='-',label='Region averages')
        plt.xlabel('Position (mm)')
        plt.ylabel('Polarity')
        plt.title(f'Frame {frame2}')
        plt.legend()

        pol_filename = os.path.basename(pol_path)
        frame_path = os.path.join(pol_direc, f'{pol_filename}.png')
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(frame_path)  # Save image with both subplots
        plt.close()

def final_frame_vel_avg(vel_path):
    with h5py.File(vel_path, "r") as f:
        dataset_names = list(f.keys())
        # Sort the dataset names numerically
        dataset_names_sorted = sorted(dataset_names, key=lambda x: int(re.search(r"\d+", x).group()))
        leng = len(dataset_names_sorted)
        # Select the middle frame and the final frame
        ff = dataset_names_sorted[leng-1]
        
        data = f[ff][:]
        velmag = np.sqrt(data[:, 0:1]**2 + data[:, 1:2]**2)
        
        return np.mean(velmag)
    
def final_frame_pol_avg(pol_path):
    with h5py.File(pol_path, "r") as f:
        dataset_names = list(f.keys())
        # Sort the dataset names numerically
        dataset_names_sorted = sorted(dataset_names, key=lambda x: int(re.search(r"\d+", x).group()))
        leng = len(dataset_names_sorted)
        # Select the middle frame and the final frame
        ff = dataset_names_sorted[leng-1]
        
        data = f[ff][:]
        polmag = np.sqrt(data[:, 0:1]**2 + data[:, 1:2]**2)
        
        return np.mean(polmag)

#%%
# import re

# def avg_x_velocity(vel_path,b,k):
#     data = []
#     with h5py.File(vel_path, "r") as f:
#         dataset_names = list(f.keys())
        
#         # Sort the dataset names numerically
#         dataset_names_sorted = sorted(dataset_names, key=lambda x: int(re.search(r"\d+", x).group()))
        
#         for name in dataset_names_sorted:
#             time = int(name)*0.1
#             xvel = f[name][:][0:1]
#             data.append([time,np.mean(xvel)])
    
#     data = np.array(data)
#     if b==0:
#         plt.subplot(3,1,1)
#         plt.plot(data[:,0], data[:,1]*1000, label=f'b={b},k={k}')
#         plt.xlabel('Time (h)')
#         plt.ylabel('Velocity (um/h)')  
#         plt.legend()
#     if b==1:
#         plt.subplot(3,1,2)
#         plt.plot(data[:,0], data[:,1]*1000, label=f'b={b},k={k}')
#         plt.xlabel('Time (h)')
#         plt.ylabel('Velocity (um/h)')  
#         plt.legend()
#     if b==5:
#         plt.subplot(3,1,3)
#         plt.plot(data[:,0], data[:,1]*1000, label=f'b={b},k={k}')
#         plt.xlabel('Time (h)')
#         plt.ylabel('Velocity (um/h)')   
#         plt.legend()

#%%
'''Calculates global average velocity and polarity'''

import mpld3

# Parameters
beta = [0,0.1,0.25,0.5,1,2,5,10,20]
kappa = [0,0.1,0.25,0.5,1,2,5,10,20]
alpha1 = [1,2,4,8,12]
alpha2 = [1,2,4,8,12]

script_directory = "C:/Users/Blitxaac/Desktop/FYP/Simulation/Results/1.1.5"
vel_direc,pol_direc = file_create(script_directory)

dic = {'Average velocity':[],
       'Average polarity':[]}

for b in beta:
    for k in kappa:
        pos = f"position_arrays_beta{b}_kappa{k}.h5"
        vel = f"velocity_arrays_beta{b}_kappa{k}.h5"
        pol = f"polarity_arrays_beta{b}_kappa{k}.h5"
# for a1 in alpha1:
#     for a2 in alpha2:
#         pos = f"position_arrays_{a1}_{a2}.h5"
#         vel = f"velocity_arrays_{a1}_{a2}.h5"
#         pol = f"polarity_arrays_{a1}_{a2}.h5"

        pos_path = os.path.join(script_directory, pos)
        vel_path = os.path.join(script_directory, vel)
        pol_path = os.path.join(script_directory, pol)
        
        if not os.path.exists(pos_path):
            continue
        
        # To initiate plotting + saving
        velplot(vel_path,pos_path,vel_direc)
        polplot(pol_path,pos_path,pol_direc)
        
        '''This section is to be used in conjuction with the next section'''
        v_avg = final_frame_vel_avg(vel_path)
        p_avg = final_frame_pol_avg(pol_path)
        
        dic['Average velocity'].append([b,k,v_avg])
        dic['Average polarity'].append([b,k,p_avg])
        # dic['Average velocity'].append([a1,a2,v_avg])
        # dic['Average polarity'].append([a1,a2,p_avg])

#%%
'''This section is used to determine the relationship between parameters'''
for key in dic:
    data = np.array(dic[key])
    k_vals = np.unique(data[:, 1])

    # Create figure
    fig = plt.figure(figsize=(16,9))

    for k in k_vals:
        mask = (data[:, 1] == k) & ~np.isnan(data[:, 2])
        x_vals = data[mask, 0]
        y_vals = data[mask, 2]
        
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', label=f'k={k}') #k or a2
    
    plt.xlabel("b") #b or a1
    plt.ylabel(f"{key}")
    plt.title(f"{key} vs b for different k values") #k or a2, b or a1
    plt.legend()
    plt.grid(False)

    plt.show()

    # Save interactive plot as an HTML file
    html_str = mpld3.fig_to_html(fig)
    # Specify a valid file path, including filename
    file_path = os.path.join(script_directory, f"{key}.html")
    
    # Write HTML content to the file
    with open(file_path, "w") as f:
        f.write(html_str)
    
#%%