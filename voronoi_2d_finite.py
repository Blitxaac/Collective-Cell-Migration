# -*- coding: utf-8 -*-
"""
Credits to @user on stackoverflow for voronoi_finite_polygons_2d
"""
import numpy as np
import math
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def order_voronoi(points,R,deg,h5_file,frame,array=True):
    
    vor = Voronoi(points)
    
    regions,vertices = voronoi_finite_polygons_2d(vor)
    
    centroid = np.zeros((len(regions),2))

    order = np.zeros((len(regions),1))

    centroid = vor.points
    #centroid = np.zeros((len(regions),2))
    '''centroid contains the coordinate of each of the Voronoi cell's centroid'''
    
    rowskip = 0
    for k,reg in enumerate(regions):        
        # cx = 0
        # cy = 0
        # ar = 0
        verts = vertices[reg]
        # size = len(reg)
        # for j in range(size):        
        #     shoe = (verts[j][0]*verts[(j+1)%size][1] - verts[(j+1)%size][0]*verts[j][1])
        #     ar += shoe/2 #area
        #     cx += (verts[j][0]+verts[(j+1)%size][0])*shoe
        #     cy += (verts[j][1]+verts[(j+1)%size][1])*shoe

        # centroid[k,0] = cx/(6*ar)
        # centroid[k,1] = cy/(6*ar)
        
        '''Shape function for p=deg'''
        denom = 0
        numer = 0
        for j,i in enumerate(range(len(reg))):
            rv = verts[i] - centroid[k,:] #get distance from center to each of the vertices
            normrv = np.linalg.norm(rv) #norm calc
            if normrv>=5*R:
                '''If it exceeds 3R, stop calculations.
                More is needed to figure out what the exact limit of the extension should be'''
                break
            denom += normrv**deg #r^order
            deltax = verts[i][0] - centroid[k,0] ### wrong
            deltay = verts[i][1] - centroid[k,1] ### wrong
            angle = math.atan2(deltay,deltax) #find phi
            '''The angle of rv with respect to the x-axis.'''
            numer += ((normrv)**deg)*complex(math.cos(2*angle), math.sin(2*angle))
            '''complex() takes the real and imaginary parts of the numerator'''
        
        if j<len(reg)-1: #if the vertices are too far apart, keep zero in the row
            rowskip+=1 #add one to note how many rows are skipped
            continue
            
        order[k,0] = abs(numer/denom)
        
    if array is True:
        avg = np.sum(order)/(len(order)-rowskip)
        voronoi_plot(regions,vertices,order,deg,h5_file,frame,avg)
    elif array is False:
        return np.sum(order)/(len(order)-rowskip)
    elif array is None:
        return np.sum(order)
        
def voronoi_plot(regions,vertices,order,deg,h5_file,frame,avg):
    # Create figure
    fig, ax = plt.subplots()

    cmap = cm.viridis  # You can choose 'plasma', 'coolwarm', etc.
    norm = Normalize(vmin=0, vmax=1)

    # Plot regions
    for i, reg in enumerate(regions):
        points = vertices[reg]  # Get the corresponding vertices
        if order[i]==0: #remove the rows with order=0 to not plot them
            continue
        polygon = Polygon(points, closed=True, facecolor=cmap(norm(order[i])))
        ax.add_patch(polygon)

    # Set limits dynamically
    # ax.set_xlim(vertices[:, 0].min() - 1, vertices[:, 0].max() + 1)
    # ax.set_ylim(vertices[:, 1].min() - 1, vertices[:, 1].max() + 1)
    ax.set_aspect('equal')  # Maintain aspect ratio

    # Add a colorbar to indicate the meaning of colors
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=f"p={deg}")
    plt.title(f'{h5_file}:frame {frame}\np={deg}_avg={avg:.5g}')

    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.grid(True)
    plt.show()