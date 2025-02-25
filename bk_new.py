''' 
The code is originally written by Dustin
Modified by Farisan (17/5/2024)
Further modified by Isaac
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
plt.switch_backend('agg')
#from voronoi_2d_finite import order_voronoi, voronoi_plot
import math
import os
import h5py
import time
#import sys
#sys.path.append('C:/Users\Blitxaac\Desktop\FYP\Simulation\Test') #change accordingly

#----------------
start_time = time.time()
#----------------

# parameters

Np = 1500 #ori: 1500
R = 10E-3 #10E-3mm = 10um (Sepulveda (2013): Cells are around 15-25µm in size, vs leader cells 50µm) #this can be the parameter to vary at the end
dt = 0.1 #0.1 (h)
cutoff_rad = 10 #10

# interaction parameters
u0 = 30E-5 #30E-5 µm^2 #2400um^2/h^2 (sepulveda 2013)
u1 = 2E-1 #2E-1        #2um^2/h^2 (sepulveda 2013)
a0 = 15E-3 #15E-3      #8um (sepulveda 2013)
a1 = 21E-3 #21E-3      #35um (sepulveda 2013)
stdev = 1.00 #1.00     #unused a.t.m
D = 0.0005

#initial density in units (mm)^(-2), since we initially confine 1500 cells within 0.2mm x 1mm:
initial_density = Np/(0.2*1) #1500/(0.2*1)

# dynamics parameter
alpha1 = 10 #10 alpha 1 = linear damping term for velocity (1.42 in Sepulveda)
alpha2 = 10 #10 alpha 2 = linear damping term for polarity 
bet = [10] #10 (beta) #coupling constant with neighbouring cells polarity (60 in Sepulveda)
tau = 1.39 #used #1.39 (tau) Sepulveda (2013) --> increasing tau increases correlations in time but weak influence on spatial extent
eta = 0.15 #0.15 (eta) #sigma in Sepulveda (2013) = noise amplitude (150um/h^2 in paper)
kap = [1] #1 (kappa) #velocity-polarity coupling constant
initial_border = 0.2 #0.2

final = 120 #120
step1 = 30 #30
step2 = 5 #5

'''Since the random number are generated in a particular sequence, 
setting it to zero yields the same set of random numbers every run'''
np.random.seed(0)


'''The Simulation class contains cell generator(including number of cells and timesteps used in the simulation),
conservation of momentum function, interaction force function, function to update the position and velocity of the cells'''

class Simulation(): 
    
    def __init__(self, Np, R, dt, cutoff_rad, initial_density, alpha1, alpha2, kappa, beta, tau, eta, initial_border, u0, u1, a0, a1, D): 
        #create instance of the class with the arguments sim=Simulation(Np,R,dt,cutoff_rad, initial_density, alpha, beta, gamma, eta, kappa, initial_border)
        
        #Cell characteristic
        self.Np = Np #initial number of cells
        self.R = R #cell radius
        self.dt = dt #time interval
        
        #2D simulation box size
        self.Xmax = initial_border*5
        self.Xmin = 0
        self.Ymax = 1
        self.Ymin = 0
        self.initial_border = initial_border
        
        #Dynamics parameters
        self.cutoff_rad = cutoff_rad #interaction length (how many cells)
        self.initial_density = initial_density
        self.alpha1 = alpha1 #damiping strength for velocity
        self.alpha2 = alpha2 #damping strength for polarity
        self.kappa = kappa #velocity-polarity coupling constant
        self.beta = beta #coupling constant
        self.tau = tau #correlation time
        self.eta = eta #sigma
        
        self.u0 = u0 #repulsion constant
        self.u1 = u1 #attraction constant
        self.a0 = a0 #repulsion length
        self.a1 = a1 #attraction length limit
        self.D = D #diffusion constant

        #Initial conditions
        self.position = np.random.uniform([self.Xmin+self.R, self.Ymin+self.R],[initial_border-self.R,self.Ymax-self.R], size =(self.Np,2)) #center of cell should not be at the border
        self.velocity = np.zeros((self.Np, 2))	
        self.deltavel = np.zeros((self.Np, 2))
        self.polarity = np.zeros((self.Np, 2))
        self.deltapol = np.zeros((self.Np, 2))
        self.local_velocity = np.zeros((self.Np,2))
        self.local_polarity = np.zeros((self.Np,2))
        self.force = np.zeros((self.Np,2))
        self.noise = np.zeros((self.Np,2))
        self.xi = np.array([np.random.normal(0,1,size=(final,2)) for m in range(self.Np)]) #delta-correlated white noise
        #self.initial_pos = np.random.uniform([self.Xmin, self.Ymin],[initial_border,self.Ymax], size =(self.Np,2))
        
        #for analysis
        self.backpairs, self.centpairs, self.fronpairs, self.pola_fron = self.tracker()
        
        self.backdist = []
        self.centdist = []
        self.frondist = []
        
        self.backvelo = []
        self.centvelo = []
        self.fronvelo = []
        
        #self.order = []
        #self.order1 = []
        
    def distcalc(self):
        '''Now, we record the distance'''
        
        for sec,pairs in enumerate([self.backpairs, self.centpairs, self.fronpairs],start=1):
            running_total = 0
            
            for pair in pairs:
                a = pair[0]
                b = pair[1]
                dist = np.linalg.norm(self.position[a] - self.position[b])
                running_total += dist
                
            mean_dist = running_total/len(pairs)
            info = [iteration1*0.1,mean_dist]
            
            if sec==1:
                self.backdist.append(info)
            elif sec==2:
                self.centdist.append(info)
            elif sec==3:
                self.frondist.append(info)

    def velocalc(self):
        '''Similar to distcalc, we record the magnitude of the velocity of the pairs'''
        for sec,pairs in enumerate([self.backpairs, self.centpairs, self.fronpairs],start=1):
            running_total = 0
            
            for pair in pairs:
                a = pair[0]
                b = pair[1]
                mag_a = np.linalg.norm(self.velocity[a])
                mag_b = np.linalg.norm(self.velocity[b])
                
                avg_mag = (mag_a+mag_b)/2
                
                running_total += avg_mag
                
            mean_velo = running_total/len(pairs)
            info = [iteration1*0.1,mean_velo]
            
            if sec==1:
                self.backvelo.append(info)
            elif sec==2:
                self.centvelo.append(info)
            elif sec==3:
                self.fronvelo.append(info)

    '''Elastic collision between cells and Xmin boundary, and periodic condition on Ymin and Ymax boundaries'''
    def collision(self): 
        for i in range(self.Np):
            x = self.position[i,0]
            y = self.position[i,1]

            '''We should not impose periodic boundary condition in the x-axis
               But we should consider the case when the cell is colliding against the Xmin boundary,
               this way we will prevent the cell's center to end up in the left side of the Xmin boundary'''
            if x < self.Xmin + self.R:
                self.position[i,0] = self.Xmin+self.R
                '''If the x-component of the velocity is pointing to the left,
                   we flip its sign so that it describes the cell being reflected'''
                if self.velocity[i,0] < 0:
                    self.velocity[i,0] = -self.velocity[i,0]    #hmm should polarity be reflected too?
            '''We should impose the periodic boundary condition in the y-axis'''
            if y > self.Ymax:    
                self.position[i,1] = self.Ymin + y - self.Ymax   
            elif y < self.Ymin:
                self.position[i,1] = self.Ymax + y - self.Ymin

    '''Functions to calculate mean border position, disorder parameter, and curve fitting'''
    def mean_border_position(self): #calculate the average x-position of the boundary cells by splitting the cells into 5 sections
        y_bin1 = [] #bin 0.0-0.2
        y_bin2 = [] #bin 0.2-0.4
        y_bin3 = [] #bin 0.4-0.6
        y_bin4 = [] #bin 0.6-0.8
        y_bin5 = [] #bin 0.8-1.0
        for i in range(self.Np):
            xpos = self.position[i,0]
            ypos = self.position[i,1]
            if 0.0*self.Ymax <= ypos < 0.2*self.Ymax:
                y_bin1.append(xpos)
            elif 0.2*self.Ymax <= ypos < 0.4*self.Ymax:
                y_bin2.append(xpos)
            elif 0.4*self.Ymax <= ypos < 0.6*self.Ymax:
                y_bin3.append(xpos)
            elif 0.6*self.Ymax <= ypos < 0.8*self.Ymax:
                y_bin4.append(xpos)
            else:
                y_bin5.append(xpos)
        
        topcells = 15 #what's the rational for this value?
        y_bin1.sort(reverse=True)
        if len(y_bin1) > topcells:
            y_bin1 = y_bin1[:topcells]
        else:
            pass
        y_bin2.sort(reverse=True)
        if len(y_bin2) > topcells:
            y_bin2 = y_bin2[:topcells]
        else:
            pass
        y_bin3.sort(reverse=True)
        if len(y_bin3) > topcells:
            y_bin3 = y_bin3[:topcells]
        else:
            pass
        y_bin4.sort(reverse=True)
        if len(y_bin4) > topcells:
            y_bin4 = y_bin4[:topcells]
        else:
            pass
        y_bin5.sort(reverse=True)
        if len(y_bin5) > topcells:
            y_bin5 = y_bin5[:topcells]
        else:
            pass
        boundary = y_bin1 + y_bin2 + y_bin3 + y_bin4 + y_bin5
        mean_border_progression = np.mean(boundary)
        
        return mean_border_progression
    
    def tracker(self):
        x_pos = np.argsort(self.position[:,0])
        num_sections = 3
        sections = np.array_split(x_pos, num_sections)
        
        #sorts the according indexes into the sections
        b_ind = sections[0]
        c_ind = sections[1]
        f_ind = sections[2] 

        bpos = self.position[b_ind]
        cpos = self.position[c_ind]
        fpos = self.position[f_ind]

        back = np.hstack((bpos,b_ind.reshape(len(b_ind),1)))
        cent = np.hstack((cpos,c_ind.reshape(len(c_ind),1)))
        fron = np.hstack((fpos,f_ind.reshape(len(f_ind),1)))
        #tempfron = np.hstack((fpos,f_ind.reshape(len(f_ind),1)))
        #temp = np.argsort(tempfron[:,0])[::-1]
        #fron = tempfron[temp][0:round(len(temp)*0.2)] #this selects the top 20% cells

        pola_fron = fron[:,2].astype(int)
        '''we will proceed to select n number of random cell pair(s) from the 3 regions'''
        rep = 20 #number of cell pairs

        bpairs = []
        cpairs = []
        fpairs = []
        
        for sec,sect in enumerate([back,cent,fron],start=1): #sect[x,y,pos_index] 
            selected = 0 #changed sec from counter to enumerate
            while selected < rep:
                #select a random cell
                sect_int = np.random.randint(len(sect))
                cell = sect[sect_int]
                
                #select the cell 2nd closest to this cell (because the closest cell is itself)
                dist = np.linalg.norm(self.position - cell[0:2], axis = 1)
                sort_dist = np.argsort(dist)
                
                cellpair = sort_dist[1]
                
                pair = [int(cell[2]),cellpair]
                
                if sec==1:
                    bpairs.append(pair)
                elif sec==2:
                    cpairs.append(pair)
                elif sec==3:
                    fpairs.append(pair)
                
                selected+=1
        
        return bpairs, cpairs, fpairs, pola_fron
    
    def increment(self):
        upperperiodic = []
        lowerperiodic = []
        velupper = []
        vellower = []
        polupper = []
        pollower = []

        for i in range(self.Np):
            if self.position[i,1] >= self.Ymax - self.cutoff_rad*self.R:
                lowerperiodic.append([self.position[i,0], self.position[i,1]-self.Ymax])
                vellower.append(self.velocity[i])
                pollower.append(self.polarity[i])
            elif self.position[i,1] <= self.Ymin + self.cutoff_rad*self.R:
                upperperiodic.append([self.position[i,0], self.position[i,1]+self.Ymax])
                velupper.append(self.velocity[i])
                polupper.append(self.polarity[i])
                
        lowerperiodic = np.array(lowerperiodic)
        upperperiodic = np.array(upperperiodic)
        
        velupper = np.array(velupper)
        vellower = np.array(vellower)
        
        polupper = np.array(polupper)
        pollower = np.array(pollower)
        
        positionperiodic = np.vstack((self.position,upperperiodic,lowerperiodic))
        velocityperiodic = np.vstack((self.velocity,velupper,vellower))
        polarityperiodic = np.vstack((self.polarity,polupper,pollower))
        
        sextants = [
            (0, math.pi/3),
            (math.pi/3, 2*math.pi/3),
            (2*math.pi/3, math.pi),
            (-math.pi, -2*math.pi/3),
            (-2*math.pi/3, -math.pi/3),
            (-math.pi/3, 0)
            ]
        
        '''The first Np rows of positionperiodic and velocityperiodic are reserved for cells within the box.
           We only care about the calculation involving the nearest neighbors of cells within the box, so:'''
        for i in range(self.Np):
            nn = np.zeros((6,9))
            nn[:] = np.nan
            '''nn is an array that will store the information regarding nearest neighbour cells in each section,
               we initialize its entries to NaN to make the calculation easier down the line'''
            #with this, we will obtain info of the 6 nearest neighbouring cells
            r1 = positionperiodic[i]
            
            cells = {f'cell_within_cutoff{c}': np.zeros((1,9)) for c in range(1, 7)}
            
            counts = {f'count{c}': 0 for c in range(1,7)}
            
            '''But since the nearest neighbors of some cells near the Ymax (and Ymin)
               boundary might be near the Ymin (and Ymax) boundary, we run all elements''' #how can I make this more efficient?
            for j in range(np.shape(positionperiodic)[0]):
                if j != i:
                    r2 = positionperiodic[j]
                    r = np.sqrt(np.dot(r1-r2,r1-r2))
                    if r <= self.cutoff_rad*self.R and r != 0:                        
                        deltax = positionperiodic[j,0] - positionperiodic[i,0]
                        deltay = positionperiodic[j,1] - positionperiodic[i,1] 
                        angle = math.atan2(deltay,deltax)
                        # v2 = velocityperiodic[j] #just uncomment this and the delpolarity below
                        v2 = polarityperiodic[j]
                        
                        '''store_info is an array that contains the
                            position (first 2 columns [0,1]),
                            velocity/polarity (columns [2,3]),
                            the distance to the i-th cell of the j-th cell (fifth column [5])'''
                            
                        '''The entries of the 6th and 7th column [6,7] are now for density purposes'''
                        store_info = np.concatenate((r2, v2, np.array([0,r,0,0,0])), axis=None)
                        '''Filtering neighbors within R=100 micrometer for each of the 6 equal sections'''
                        
                        for s in range(1,7):
                            if angle >= sextants[s-1][0] and angle < sextants[s-1][1]:
                                key = f'cell_within_cutoff{s}'
                                cou = f'count{s}'
                                # Append `store_info` as a new row
                                cells[key] = np.vstack((cells[key], store_info.reshape(1,9)))
                                counts[cou]+=1

            '''For each section, we take only the nearest neighbor'''
            for cwc in range(1,7):
                cell_within_cutoff = cells[f'cell_within_cutoff{cwc}'][1:]
                count = counts[f'count{cwc}']
                if count != 0:
                    indexmin = np.argmin(cell_within_cutoff[:,5])
                    nn[(cwc-1)] = cell_within_cutoff[indexmin]

            ###########################TBC###########################
            tot_count = counts['count1'] + counts['count2'] + counts['count3'] + counts['count4'] + counts['count5'] + counts['count6']
            loc_area = math.pi*(self.R*self.cutoff_rad)**2
            loc_dens = tot_count/loc_area
            
            # mean_dist = np.nanmean(nn[:,5])
            # loc_dens = 4/(math.pi*mean_dist**2)
            ###########################TBC###########################

            '''This section, we will calculate the density and density vector (x,y) of each sextant
            and append it to nn[:,6], and nn[:,7] & nn[:,8] respectively'''
            # mean_dist = np.nanmean(nn[:,5])
            for sext in range(1,7):
                # Density
                count = counts[f'count{sext}']
                sextant_area = math.pi*(cutoff_rad*R)**2/6
                sextant_dens = count/sextant_area
                nn[(sext-1),6] = sextant_dens
                
                # Density vector
                if count==0:
                    nn[(sext-1),7] = 0
                    nn[(sext-1),8] = 0
                    continue
                key = f'cell_within_cutoff{sext}'
                # 100um radius
                vectors = cells[key][1:,0:2] - r1.reshape(1,-1)
                # Normalize
                magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
                unit_vectors = vectors / magnitudes
                avg_vector = np.mean(unit_vectors, axis=0)
                avg_direction = avg_vector / np.linalg.norm(avg_vector)
                nn[(sext-1),7] = avg_direction[0] # x-component
                nn[(sext-1),8] = avg_direction[1] # y-component
                
                # mean_dist radius
                # cells1 = cells[key][cells[key][:, 5] < mean_dist]
                # vectors = cells1[1:,0:2] - r1.reshape(1,-1)
                # if len(vectors)==0:
                #     nn[(sext-1),7] = 0
                #     nn[(sext-1),8] = 0
                #     continue
                # magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
                # unit_vectors = vectors / magnitudes
                # avg_vector = np.mean(unit_vectors, axis=0)
                # avg_direction = avg_vector / np.linalg.norm(avg_vector)
                # nn[(sext-1),7] = avg_direction[0] # x-component
                # nn[(sext-1),8] = avg_direction[1] # y-component
            
            '''Density dependent diffusion''' 
            # The vector should point in the opposite direction, from high to low density, w.r.t. the density of the regions
            dens_norm = nn[:,6]/np.sum(nn[:,6])
            dens_vector = (-np.sum(dens_norm*nn[:,7]),-np.sum(dens_norm*nn[:,8])) 


            '''The entries for sections not containing nearest neighbors (nn) will all be NaN,
               so we calculate the mean velocity with numpy's nanmean that ignores NaN entries'''
            self.local_polarity[i] = np.array([np.nanmean(nn[:,2]), np.nanmean(nn[:,3])])
            #self.local_velocity[i] = np.array([np.nanmean(nn[:,2]), np.nanmean(nn[:,3])]) #as seen in store_info
            
            '''Force calculation due to repulsions and attractions'''
            force_cluster = np.zeros(2)
            for k in range(6):     
                if np.isnan(nn[k]).any() == False:
                    dir_vec = np.array([nn[k,0],nn[k,1]]) - r1
                    dir_vec /= nn[k,5]
                    force_magnitude_repulsive = self.u0*abs(-2*nn[k,5]*math.exp(-nn[k,5]*nn[k,5]/(self.a0*self.a0))/(self.a0*self.a0))
                    if nn[k,5] > self.a1:
                        force_magnitude_attractive = self.u1*2*(nn[k,5]-self.a1)
                    else:
                        force_magnitude_attractive = 0
                    #repulsive force is always pointing in the opposite direction of r_j-r_i, while attractive force is pointing at the dirrection of r_j-r_i
                    force_cluster += (-force_magnitude_repulsive + force_magnitude_attractive)*dir_vec            

            '''Increments'''
            self.noise[i] += (self.xi[i,iteration1-1]-self.noise[i])*self.dt/self.tau
            self.force[i] = force_cluster
            self.deltapol[i] = self.dt*(self.eta*self.noise[i] + self.beta*(self.local_polarity[i]-self.polarity[i]) - self.alpha2*self.polarity[i]) + self.D*dens_vector
            #self.deltapol[i] = self.dt*(self.eta*self.noise[i] + self.beta*(self.local_velocity[i]-self.velocity[i]) - self.alpha2*self.polarity[i])
            self.deltavel[i] = self.dt*(self.kappa*self.polarity[i] - self.alpha1*self.velocity[i])+self.force[i]
            self.polarity[i] += self.deltapol[i]
            self.velocity[i] += self.deltavel[i]
            self.position[i] += self.dt*self.velocity[i]
            
            '''Cell division'''
            if self.initial_density >= loc_dens:
                if self.position[i,0] < self.mean_border_position()-(2*self.R*3) and self.position[i,0] >= self.Xmin+self.R:   
                    #we need to make sure the position of the new cells do not overlap with any of the old cells', otherwise we will encounter error when calculating the force (can't divide by zero distance)
                    #for this reason we implement a do-while loop, which executes at least once. The loop will be terminated only when the new cell does not overlap with old cells
                    while True:
                        polar_angle = np.random.uniform(0, 2*math.pi)
                        x = 2*self.R*math.cos(polar_angle)
                        y = 2*self.R*math.sin(polar_angle)
                        new_xlocation = self.position[i,0] + x
                        new_ylocation = self.position[i,1] - y
                        #for the right boundary of the bulk
                        if self.position[i,0] + x > self.mean_border_position()-(2*self.R*3):
                            #this occurs only when x is positive, so we need to transform x to -x to keep cell divisions within the bulk
                            new_xlocation = self.position[i,0] - x

                        #for the left boundary of the bulk
                        elif self.position[i,0] + x < self.Xmin+self.R:
                            #this occurs only when x is negative, so we need to transform -|x| to |x| to keep cell divisions within the bulk
                            new_xlocation = self.position[i,0] - x
                            #as can be seen, both conditions share the same expression, so we can just write both conditions in the if statement with "or" condition

                        #for the upper boundary of the box
                        if self.position[i,1] + y > self.Ymax:
                            #y must be positive for this to happen, since position[i,1] is always smaller than or equal to Ymax
                            new_ylocation = self.position[i,1] + y - self.Ymax
                            #periodic
                        #for the lower boundary of the box
                        elif self.position[i,1] + y < self.Ymin:
                            #y must be negative for this to happen, since position[i,1] is always larger than or equal to Ymin
                            new_ylocation = self.Ymax - abs(self.position[i,1] + y)                  
                        if (np.array([new_xlocation, new_ylocation]) == self.position).all(axis = 1).any() == False:
                            break
                        
                    new_pos = np.array([new_xlocation, new_ylocation])
                    self.position = np.vstack((self.position, new_pos))
                    #self.initial_pos = np.vstack((self.initial_pos, new_pos))
                    self.velocity = np.vstack((self.velocity, np.zeros(2)))
                    self.deltavel = np.vstack((self.deltavel, np.zeros(2)))
                    self.polarity = np.vstack((self.polarity, np.zeros(2)))
                    self.deltapol = np.vstack((self.deltapol, np.zeros(2)))
                    self.local_velocity = np.vstack((self.local_velocity, np.zeros(2)))
                    self.local_polarity = np.vstack((self.local_polarity, np.zeros(2)))
                    self.force = np.vstack((self.force, np.zeros(2)))
                    self.noise = np.vstack((self.noise, np.zeros(2)))
                    self.xi = np.append(self.xi, [np.append(np.zeros((iteration1-1,2)), np.random.normal(0,1,size=(final-(iteration1-1),2)), axis = 0)], axis = 0)                      
                    self.Np = self.Np + 1
        
        self.distcalc()
        self.velocalc()
        
        #orderp2 = order_voronoi(self.position,R,array=False)
        #self.order.append([iteration1*0.1,orderp2])
        
        #orderp21 = order_voronoi(self.position,R,array=None)
        #self.order1.append([iteration1*0.1,orderp21])
        
        self.collision()

    def order_parameter(self):
        v =  self.velocity
        
        norm = np.linalg.norm(v, axis=1)

        np.place(norm,norm==0, 1)
        new_norm = np.column_stack((norm,norm))

        normalised_v = v/new_norm
        sum_velocity = np.sum(normalised_v, axis=0)
        abs_sum = np.linalg.norm(sum_velocity)
        order = abs_sum/self.Np
        return order
    
    def correlationvp(self):

        v=self.velocity
        p=self.polarity
        
        norm_v = np.linalg.norm(v, axis=1)
        np.place(norm_v,norm_v==0, 1)
        new_norm_v = np.column_stack((norm_v,norm_v))
        normalised_v = v/new_norm_v


        norm_p = np.linalg.norm(p, axis=1)
        np.place(norm_p,norm_p==0, 1)
        new_norm_p = np.column_stack((norm_p,norm_p))
        normalised_p = p/new_norm_p


        dot = normalised_v[:,0]*normalised_p[:,0] + normalised_v[:,1]*normalised_p[:,1]
        # element = np.count_nonzero(dot)
        # if element == 0:
        #     correlation = 0
        # else:
        #     correlation =np.sum(dot)/element
        correlation = np.mean(dot)
        
        return correlation

'''This section is used for recording position data'''
def initialize_hdf5_file(file_path):
    with h5py.File(file_path, "w") as f:
        # Optionally add metadata or groups
        f.attrs["description"] = "Position data from iterations"

def append_position_data(file_path, iteration, position_array):
    with h5py.File(file_path, "a") as f:  # Open in append mode
        dataset_name = f"{iteration}"
        f.create_dataset(dataset_name, data=position_array, chunks=True, compression="gzip")

'''This section is used for recording velocity data'''

def initialize_hdf5_file_vel(file_path):
    with h5py.File(file_path, "w") as f:
        # Optionally add metadata or groups
        f.attrs["description"] = "Velocity data from iterations"

def append_velocity_data(file_path, iteration, velocity_array):
    with h5py.File(file_path, "a") as f:  # Open in append mode
        dataset_name = f"{iteration}"
        f.create_dataset(dataset_name, data=velocity_array, chunks=True, compression="gzip")

'''This section is used for recording polarity data'''

def initialize_hdf5_file_pol(file_path):
    with h5py.File(file_path, "w") as f:
        # Optionally add metadata or groups
        f.attrs["description"] = "Polarity data from iterations"

def append_polarity_data(file_path, iteration, polarity_array):
    with h5py.File(file_path, "a") as f:  # Open in append mode
        dataset_name = f"{iteration}"
        f.create_dataset(dataset_name, data=polarity_array, chunks=True, compression="gzip")

#----------------------------------------------------------------------
for b in bet: #change
    for k in kap: #change
        iteration1 = 0
        border = []
        order = []
        velocity = []
        polarity = []
        correlation = []
        meansqdisp = []
        avglocaldensity = []
        avgglobaldensity = []
        
#----------------------------------------------------------------------
        script_directory = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()

        hdf5_file_path = os.path.join(script_directory, f"position_arrays_beta{b}_kappa{k}.h5")
        if not os.path.exists(hdf5_file_path):
            initialize_hdf5_file(hdf5_file_path)
            
        hdf5_file_path_vel = os.path.join(script_directory, f"velocity_arrays_beta{b}_kappa{k}.h5")
        if not os.path.exists(hdf5_file_path_vel):
            initialize_hdf5_file_vel(hdf5_file_path_vel)

        hdf5_file_path_pol = os.path.join(script_directory, f"polarity_arrays_beta{b}_kappa{k}.h5")
        if not os.path.exists(hdf5_file_path_pol):
            initialize_hdf5_file_pol(hdf5_file_path_pol)
#----------------------------------------------------------------------
    
        #To vary
        beta = b
        kappa = k
        
        sim = Simulation(Np, R, dt, cutoff_rad, initial_density, alpha1, alpha2, kappa, beta, tau, eta, initial_border, u0, u1, a0, a1) #assigning the class to a variable so the functions inside can be accessed
        while iteration1 <= final:
            
            if iteration1 > 0:
                sim.increment()
                append_position_data(hdf5_file_path, iteration1, sim.position) #position file
                append_velocity_data(hdf5_file_path_vel, iteration1, sim.velocity) #velocity file
                append_polarity_data(hdf5_file_path_pol, iteration1, sim.polarity) #polarity file
                
                
            # if iteration1 % step1 == 0:
            #     r=sim.position
            #     np.savetxt('Position{}.csv'.format(iteration1), r, delimiter=',', fmt='%s')
            #     v=sim.velocity
            #     np.savetxt('Velocity{}.csv'.format(iteration1), v, delimiter=',', fmt='%s')
            #     p=sim.polarity
            #     np.savetxt('Polarity{}.csv'.format(iteration1), p, delimiter=',', fmt='%s')
            
            #     magvel = np.linalg.norm(v, axis=1)
            #     magpol = np.linalg.norm(p, axis=1)
            
            #     plt.scatter(sim.position[:,0],sim.position[:,1],color="red", s=2)
                
            #     plt.xlim(0, sim.Xmax)
            #     plt.ylim(0, sim.Ymax)
            #     plt.savefig("Config{}.pdf".format(iteration1))
            #     plt.close()
        
            #     norm = Normalize()
            #     norm.autoscale(magvel)
            #     colormap = cm.viridis_r
            #     sm = cm.ScalarMappable(cmap=colormap, norm=norm)
            #     sm.set_array([])
            #     norm2 = Normalize()
            #     norm2.autoscale(magpol)
            #     colormap2 = cm.inferno_r
            #     sm2 = cm.ScalarMappable(cmap=colormap2, norm=norm2)
            #     sm2.set_array([])
            #     if iteration1 > 0:
                    
            #         plt.quiver(sim.position[:,0],sim.position[:,1], sim.velocity[:,0],sim.velocity[:,1], color=colormap(norm(magvel)))
            #         plt.xlim(0, sim.Xmax)
            #         plt.ylim(0, sim.Ymax)
                    
            #         plt.colorbar(sm,ax=plt.gca())
            #         plt.savefig("Velocity{}.pdf".format(iteration1))
            #         plt.close()
        
            #         plt.quiver(sim.position[:,0],sim.position[:,1], sim.polarity[:,0],sim.polarity[:,1],color=colormap2(norm2(magpol[:]))) 
            #         plt.xlim(0, sim.Xmax)
            #         plt.ylim(0, sim.Ymax)
                    
            #         plt.colorbar(sm2,ax=plt.gca())
            #         plt.savefig("Polarity{}.pdf".format(iteration1))
            #         plt.clf()
          
            # if iteration1 % step2 == 0:
            #     f = open('MeanBorder.csv', 'a')
            #     f.write('%.10f\n' % sim.mean_border_position())
            #     f.close()
            #     f = open('Order.csv', 'a')
            #     f.write('%.10f\n' % sim.order_parameter())
            #     f.close()
            #     f = open('Correlationvp.csv', 'a')
            #     f.write('%.10f\n' % sim.correlationvp())
            #     f.close()
            #     f = open('CellNumber.txt', 'a')
            #     f.write('%d\n' % sim.Np)
            #     f.close()      
            
            iteration1 += 1
        
        # f = open('Details.csv', 'a')
        # f.write('Total cells: %d\nRepulsive strength: %.10f\nAttractive strength: %.10f\nRepulsive distance: %.10f\nAttractive distance: %.10f\nBeta: %.10f\n' % (sim.Np,sim.u0,sim.u1,sim.a0,sim.a1,sim.beta))
        # f.close()
        
        # This section is used for measuring the change in cell-cell distances over time
        plt.figure()
        ddata_back = np.array(sim.backdist)
        ddata_cent = np.array(sim.centdist)
        ddata_fron = np.array(sim.frondist)
        plt.plot(ddata_back[:,0],ddata_back[:,1],label='back')
        plt.plot(ddata_cent[:,0],ddata_cent[:,1],label='center')
        plt.plot(ddata_fron[:,0],ddata_fron[:,1],label='front')
        plt.xlabel("Time(h)")
        plt.ylabel("Average Distance(mm)")
        plt.legend()
        plt.savefig(f"cellpair dist avg,b={b},k={k}.png", dpi=300, bbox_inches="tight")
        
        # # And then this section is for velocity calculations
        plt.figure()
        ddata_back = np.array(sim.backvelo)
        ddata_cent = np.array(sim.centvelo)
        ddata_fron = np.array(sim.fronvelo)
        plt.plot(ddata_back[:,0],ddata_back[:,1],label='back')
        plt.plot(ddata_cent[:,0],ddata_cent[:,1],label='center')
        plt.plot(ddata_fron[:,0],ddata_fron[:,1],label='front')
        plt.xlabel("Time(h)")
        plt.ylabel("Average Velocity")
        plt.legend()
        plt.savefig(f"velocity avg,b={b},k={k}.png", dpi=300, bbox_inches="tight")
        
        # # Average p=2 shape function
        # data = np.array(sim.order)
        # fig1, ax1 = plt.subplots()
        # ax1.plot(data[:,0],data[:,1],label='p=2')
        # ax1.xlabel("Time(h)")
        # ax1.ylabel("Average p=2 shape function")
        # ax1.legend()
        # fig1.savefig("p=2 avg.png", dpi=300, bbox_inches="tight")
        
        # # Total p=2 shape function
        # data1 = np.array(sim.order1)
        # fig2, ax2 = plt.subplots()
        # ax2.plot(data1[:,0],data1[:,1],label='p=2')
        # ax2.xlabel("Time(h)")
        # ax2.ylabel("Total p=2 shape function")
        # ax2.legend()
        # fig2.savefig("p=2 total.png", dpi=300, bbox_inches="tight")

end_time = time.time()
time_taken = int(end_time - start_time)
minutes, seconds = divmod(time_taken,60)
formatted_time = f"{minutes:02}:{seconds:02}"
print(f'The time taken is {formatted_time}')
