# Copyright (C) 2025  Helmholtz-Zentrum-Hereon

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

"""
libraries for simulation

Created on Fri Jun 23 10:28:09 2023

@author: Arnab Majumdar
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
from scipy.special import erf
import h5py

def model_run(sim_model, nodes, midpoint, t, t_end):
    """
    Function desc:
    a) Choosing which model to run
    b) Assign sld to mesh : sim_sld
    c) max and min sld for cbar in gif : sld_max, sld_min
    """
    if sim_model=='ball':
        sim_sld, sld_max, sld_min = model_ball(nodes, midpoint)
    if sim_model=='box':
        sim_sld, sld_max, sld_min = model_box(nodes)
    if sim_model=='bib':
        sim_sld, sld_max, sld_min = model_bib(nodes, midpoint)
    if sim_model=='bib_ecc':
        sim_sld, sld_max, sld_min = model_bib_ecc(nodes, midpoint)
    if sim_model=='gg':
        sim_sld, sld_max, sld_min = model_gg(nodes, midpoint, t, t_end)
    if sim_model=='fs':
        sim_sld, sld_max, sld_min = model_fs(nodes, midpoint, t, t_end)
    if sim_model=='sld_grow':
        sim_sld, sld_max, sld_min = model_sld_grow(nodes, midpoint, t, t_end)
    if sim_model=='phase_field':
        sim_sld, sld_max, sld_min = model_phase_field()
    if sim_model=='chemevo':
        sim_sld, sld_max, sld_min = model_chemevo(nodes, midpoint, t, t_end)
    if sim_model=='probsimu':
        sim_sld, sld_max, sld_min = model_probsimu(nodes, midpoint, t, t_end)
    return sim_sld, sld_max, sld_min

def model_phase_field():
    """
    Function desc:
    define sld, man and min for phase field model 
    """
    # read model_run_param from xml
    xml_folder='./xml/'
    model_xml=os.path.join(xml_folder, 'model_phase_field.xml')
    tree=ET.parse(model_xml)
    root = tree.getroot()
    # read params
    name=root.find('name').text
    time=float(root.find('time').text)
    # run simulation
    time_int=int(time)
    file_name=f'moose/{name}_{time_int:0>5}.h5'
    file=h5py.File(file_name,'r')
    sim_sld=file['SLD'][:]
    sld_max=np.max(sim_sld)
    sld_min=np.min(sim_sld)
    return sim_sld, sld_max, sld_min


def model_ball(nodes, midpoint):
    """
    Function desc:
    define sld, man and min for ball model 
    """
    # read model_run_param from xml
    xml_folder='./xml/'
    struct_xml=os.path.join(xml_folder, 'model_ball.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # read params
    rad=float(root.find('rad').text)
    sld=float(root.find('sld').text)
    # run simulation
    nodes = np.array(nodes)
    sim_sld = np.zeros(len(nodes))
    cord_ed = np.sum((nodes-midpoint)**2,axis=1)
    sim_sld [cord_ed <= rad**2] = sld
    sld_max=sld
    sld_min=0
    return sim_sld, sld_max, sld_min

def model_box(nodes):
    """
    Function desc:
    define sld, man and min for box model 
    """
    # read model_run_param from xml
    xml_folder='./xml/'
    struct_xml=os.path.join(xml_folder, 'model_box.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # read params
    sld=float(root.find('sld').text)
    # run simulation
    nodes = np.array(nodes)
    sim_sld = sld*np.ones(len(nodes))
    sld_max=sld
    sld_min=0
    return sim_sld, sld_max, sld_min

def model_bib(nodes, midpoint):
    """
    Function desc:
    define sld, man and min for bib model 
    """
    # read model_run_param from xml
    xml_folder='./xml/'
    struct_xml=os.path.join(xml_folder, 'model_bib.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # read params
    rad=float(root.find('rad').text)
    sld_ball=float(root.find('sld_in').text)
    sld_box=float(root.find('sld_out').text)
    # run simulation
    nodes = np.array(nodes)
    sim_sld = sld_box*np.ones(len(nodes))
    cord_ed = np.sum((nodes-midpoint)**2,axis=1)
    sim_sld [cord_ed <= rad**2] = sld_ball
    sld_max=sld_ball
    sld_min=sld_box
    return sim_sld, sld_max, sld_min

def model_bib_ecc(nodes, midpoint):
    """
    Function desc:
    define sld, man and min for bib_ecc model 
    """
    # read model_run_param from xml
    xml_folder='./xml/'
    struct_xml=os.path.join(xml_folder, 'model_bib_ecc.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # read params
    rad=float(root.find('rad').text)
    sld_ball=float(root.find('sld_in').text)
    sld_box=float(root.find('sld_out').text)
    ecc_x=float(root.find('ecc').find('x').text)
    ecc_y=float(root.find('ecc').find('y').text)
    ecc_z=float(root.find('ecc').find('z').text)
    ecc=np.array([ecc_x, ecc_y, ecc_z])
    # run simulation
    nodes = np.array(nodes)
    sim_sld = sld_box*np.ones(len(nodes))
    ball_mid=midpoint+ecc
    cord_ed = np.sum((nodes-ball_mid)**2,axis=1)
    sim_sld [cord_ed <= rad**2] = sld_ball
    sld_max=sld_ball
    sld_min=sld_box
    return sim_sld, sld_max, sld_min

def model_gg(nodes, midpoint, t, t_end):
    """
    Function desc:
    define sld, man and min for gg model 
    """
    # read model_run_param from xml
    xml_folder='./xml/'
    struct_xml=os.path.join(xml_folder, 'model_gg.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # read params
    rad_0=float(root.find('rad_0').text)
    rad_end=float(root.find('rad_end').text)
    sld_grain=float(root.find('sld_in').text)
    sld_env=float(root.find('sld_out').text)
    # run simulation
    nodes = np.array(nodes)
    sim_sld = sld_env*np.ones(len(nodes))
    rad=rad_0+t*(rad_end-rad_0)/t_end
    cord_ed = np.sum((nodes-midpoint)**2,axis=1)
    sim_sld [cord_ed <= rad**2] = sld_grain
    sld_max=sld_grain
    sld_min=sld_env
    return sim_sld, sld_max, sld_min

def model_fs(nodes, midpoint, t, t_end):
    """
    Function desc:
    define sld, man and min for fs model 
    """
    # read model_run_param from xml
    xml_folder='./xml/'
    struct_xml=os.path.join(xml_folder, 'model_fs.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # read params
    rad=float(root.find('rad').text)
    sig_0=float(root.find('sig_0').text)
    sig_end=float(root.find('sig_end').text)
    sld_grain=float(root.find('sld_in').text)
    sld_env=float(root.find('sld_out').text)
    # run simulation
    nodes = np.array(nodes)
    sig=sig_0+t*(sig_end-sig_0)/t_end
    sim_sld = np.ones(len(nodes))
    coord_r=np.sqrt(np.sum((nodes-midpoint)**2,axis=1))
    if sig==0:
        sim_sld=(sld_env-sld_grain)*np.heaviside(coord_r-rad,0)+sld_grain
    else:
        #sld=-np.heaviside(coord_r-r,0)+1
        sim_sld=((sld_grain-sld_env)/2)*(1-erf((coord_r-rad)/(np.sqrt(2)*sig)))+sld_env
    sld_max=sld_grain
    sld_min=sld_env
    return sim_sld, sld_max, sld_min

def model_sld_grow(nodes, midpoint, t, t_end):
    """
    Function desc:
    define sld, man and min for sld_grow model 
    """
    # read model_run_param from xml
    xml_folder='./xml/'
    struct_xml=os.path.join(xml_folder, 'model_sld_grow.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # read params
    rad=float(root.find('rad').text)
    sld_start=float(root.find('sld_in_0').text)
    sld_end=float(root.find('sld_in_end').text)
    sld_env=float(root.find('sld_out').text)
    # run simulation
    nodes = np.array(nodes)
    sim_sld = sld_env*np.ones(len(nodes))
    sld_grain=sld_start+t*(sld_end-sld_start)/t_end
    cord_ed = np.sum((nodes-midpoint)**2,axis=1)
    sim_sld [cord_ed <= rad**2] = sld_grain
    sld_max=max(sld_start, sld_end, sld_env)
    sld_min=min(sld_start, sld_end, sld_env)
    return sim_sld, sld_max, sld_min

def model_chemevo(nodes, midpoint, t, t_end):
    """
    Function desc:
    define sld, man and min for chem_evo model 
    """
    # read model_run_param from xml
    xml_folder='./xml/'
    struct_xml=os.path.join(xml_folder, 'model_chemevo.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # read params
    mode=root.find('mode').text
    rad=float(root.find('rad').text)
    sld_start=float(root.find('sld_start').text)
    sld_end=float(root.find('sld_end').text)
    sld_env=float(root.find('sld_env').text)
    # run simulation
    nodes = np.array(nodes)
    cord_ed = np.sum((nodes-midpoint)**2,axis=1)
    sim_sld = sld_env*np.ones(len(nodes))
    if mode=='homo':
        sld_grain=sld_start+t*(sld_end-sld_start)/t_end
        sim_sld [cord_ed <= rad**2] = sld_grain
        sld_max=max(sld_start, sld_end, sld_env)
        sld_min=min(sld_start, sld_end, sld_env)
    elif mode=='step':
        rad_out=rad
        rad_in=rad-t*(rad/t_end)
        rad_in=max(rad_in,0)
        sim_sld [cord_ed <= rad_out**2] = sld_end
        sim_sld [cord_ed <= rad_in**2] = sld_start
        sld_max=max(sld_start, sld_end, sld_env)
        sld_min=min(sld_start, sld_end, sld_env)
    elif mode=='diffuse':
        n=50 # num_term in series
        D_coeff=float(root.find('d_coeff').text)
        r=np.sqrt(cord_ed)
        a=rad
        c1= sld_start
        c0= sld_end
        if t==0:
            sim_sld [r**2 <= a**2] = c1
        else:
            if D_coeff*t<=0.1:
                for i, _ in enumerate(r):
                    series=0
                    print(type(r))
                    if r[i] <=rad:
                        if r[i]==0:
                            for j in range(1,n):
                                series+=((-1)**j) * np.exp(-((D_coeff*(j**2)*(np.pi**2)*t)/(a**2)))
                        
                            sim_sld[i]=c1+(c0-c1)*(1+2*series)
                        else:
                            for j in range(n):
                                #series+=((-1)**j/j) * np.sin(j*np.pi*r[i]/a) * np.exp(-((D_coeff*(j**2)*(np.pi**2)*t)/(a**2)))
                                term_1=(2*j+1)*a
                                term_2=2* np.sqrt(D_coeff*t)
                                series+=erf((term_1-r[i])/term_2)-erf((term_1+r[i])/term_2)
                            sim_sld[i]=c1+(c0-c1)*(a/r[i]*series)
            else:
                for i, _ in enumerate(r):
                    series=0
                    # print(i)
                    # print(r[i])
                    if r[i] <=rad:
                        if r[i]==0:
                            for j in range(1,n):
                                series+=((-1)**j) * np.exp(-((D_coeff*(j**2)*(np.pi**2)*t)/(a**2)))                       
                            sim_sld[i]=c1+(c0-c1)*(1+2*series)
                        else:
                            for j in range(1,n):
                                series+=((-1)**j/j) * np.sin(j*np.pi*r[i]/a) * np.exp(-((D_coeff*(j**2)*(np.pi**2)*t)/(a**2)))                      
                            sim_sld[i]=c1+(c0-c1)*(1+((2*a)/(np.pi*r[i]))*series)
        sld_max=max(sld_start, sld_end, sld_env)
        sld_min=min(sld_start, sld_end, sld_env)
    else:
        print('problem with mode in chem evo xml')
    return sim_sld, sld_max, sld_min

def model_probsimu(nodes, midpoint, t, t_end):
    """
    Function desc:
    define sld, man and min for prob simu model 
    """
    # read ctruct_gen_param from xml
    xml_folder='./xml/'
    struct_xml=os.path.join(xml_folder, 'struct.xml')
    tree=ET.parse(struct_xml)
    root = tree.getroot()
    # length_a=float(root.find('lengths/x').text)
    # length_b=float(root.find('lengths/y').text)
    # length_c=float(root.find('lengths/z').text)
    nx=int(root.find('num_cell/x').text)
    ny=int(root.find('num_cell/y').text)
    nz=int(root.find('num_cell/z').text)
    # read model_run_param from xml
    model_xml=os.path.join(xml_folder, 'model_probsimu.xml')
    tree=ET.parse(model_xml)
    root = tree.getroot()
    # read params
    coarsen=int(root.find('coarsen').text)
    hd=float(root.find('hd').text)
    mr_gas=float(root.find('mr_gas').text)
    n_reac=int(root.find('num_reac').text)
    n_reag=int(root.find('num_reag').text)
    time_model=int(root.find('time_model').text)
    time_cons=float(root.find('time_cons').text)
    ## check if there are more reactant entries
    all_reacs = [child for child in root if child.tag.startswith("reac_")]
    if len(all_reacs) != n_reac:
        raise ValueError(
            "Mismatch: number of reactant entries are not equal to num_reac in xml file."
        )
    reac_sld_arr=np.zeros(n_reac)
    reac_mr_arr=np.zeros(n_reac)
    # read reactant entries
    for reac_idx in range(n_reac):
        reac = root.find(f"reac_{reac_idx+1}")
        if reac is None:
            raise ValueError(f"Missing entry: reac_{reac_idx+1}")
        else:
            reac_sld_arr[reac_idx]=float(reac.find("sld").text)
            reac_mr_arr[reac_idx]=float(reac.find("mr").text)
    ## check if there are more reagent entries
    all_reags = [child for child in root if child.tag.startswith("reag_")]
    if len(all_reags) != n_reag:
        raise ValueError(
            "Mismatch: number of reagent entries are not equal to num_reag in xml file."
        )
    reag_sld_arr=np.zeros(n_reag)
    reag_mr_arr=np.zeros(n_reag)
    # read reagent entries
    for reag_idx in range(n_reag):
        reag = root.find(f"reag_{reag_idx+1}")
        if reag is None:
            raise ValueError(f"Missing entry: reag_{reag_idx+1}")
        else:
            reag_sld_arr[reag_idx]=float(reag.find("sld").text)
            reag_mr_arr[reag_idx]=float(reag.find("mr").text)

    # adjust mr acc to hd and mr gas
    reac_mr_arr, reag_mr_arr = adjust_hd(reac_mr_arr, reag_mr_arr,hd)
    reag_mr_arr = adjust_mr_gas(reag_mr_arr,mr_gas)
    # calculation of chem percentage (takes time model into account)
    if time_model==0:
        # zeroth
        chem_percentage=(1/t_end)*t
    elif time_model==1:
        # first
        chem_percentage=1-np.exp(-time_cons*t)
    elif time_model==2:
        chem_percentage=(time_cons*t)/(1+time_cons*t)
    else:
        raise ValueError(
            "Time model not mentioned correctly."
        )
    # prob dis of sld
    sim_sld = np.ones(len(nodes))
    sim_sld = prob_dist(nx, ny, nz, reac_mr_arr, reag_mr_arr,
                         reac_sld_arr, reag_sld_arr,
                           chem_percentage, coarsen)
    # calculate max min
    sld_max=max(np.max(reac_sld_arr), np.max(reag_sld_arr),0)
    sld_min=min(np.min(reac_sld_arr), np.min(reag_sld_arr),0)
    return sim_sld, sld_max, sld_min

def adjust_hd(reac_mr_arr, reag_mr_arr,hd):
    """
    Function for adjusting mr acc to hd value
    only if ft-it is included while inputing
    """
    if len(reac_mr_arr) == 6 and len(reag_mr_arr) == 9:
        #hd = P_bd4
        reac_mr_arr[4]=reac_mr_arr[4]*(1-hd)
        reac_mr_arr[5]=reac_mr_arr[5]*hd
        reag_mr_arr[2]=reag_mr_arr[2]*(1-hd)
        reag_mr_arr[3]=reag_mr_arr[3]*(1-hd)
        reag_mr_arr[4]=reag_mr_arr[4]*(1-hd)
        reag_mr_arr[5]=reag_mr_arr[5]*hd
        reag_mr_arr[6]=reag_mr_arr[6]*hd
        reag_mr_arr[7]=reag_mr_arr[7]*hd
    elif len(reac_mr_arr) == 3 and len(reag_mr_arr) == 3:
        print('FT-IR is not included')
        print('Not doing hd exchange')
    else:
        print('Num of reactants does not match 6')
        raise ValueError(f"Num of [reactants, reagents] has to be [3, 3] (no ft-ir)\n"
                         f"or [6, 9] (with ft-ir),\n"
                         f"not [{len(reac_mr_arr)}, {len(reag_mr_arr)}]")
    # # adust reagents
    # if len(reag_mr_arr) == 9:
    #     #hd = P_bd4
    #     reag_mr_arr[2]=reag_mr_arr[2]*(1-hd)
    #     reag_mr_arr[3]=reag_mr_arr[3]*(1-hd)
    #     reag_mr_arr[4]=reag_mr_arr[4]*(1-hd)
    #     reag_mr_arr[5]=reag_mr_arr[5]*hd
    #     reag_mr_arr[6]=reag_mr_arr[6]*hd
    #     reag_mr_arr[7]=reag_mr_arr[7]*hd
    # elif len(reag_mr_arr) == 3:
    #     print('FT-IR is not included')
    #     print('Not doing hd exchange')
    # else:
    #     print('Num of reactants does not match 9')
    #     raise ValueError(f"Num of reactants has to be 3 (no ft-ir) or 6 (with ft-ir), not {len(reac_mr_arr)}")
    return reac_mr_arr, reag_mr_arr

def adjust_mr_gas(reag_mr_arr,mr_gas):
    """
    Function for adjusting mr acc to trapped gas quan
    """
    # adust molar ratio of trapped gas
    reag_mr_arr[-1]=mr_gas    
    return reag_mr_arr

def prob_dist(nx, ny, nz, mr_reac_arr, mr_reag_arr,
               sld_reac_arr, sld_reag_arr,
                 chem_percentage, coarsen):
    """
    Function for calculating probabilitic distribution

    """

    # calculate num coarse grains in x, y, z direction
    # also check if some of the grain are partially outside sim box
    # e.g. if len is 20 coarsen is 10
    # there are two grains
    # if coarsen is 11
    # then second grain is only partially fitted inside length
    if (nx+1)%coarsen==0:
        coarsen_nx=(nx+1)//coarsen
    else:
        coarsen_nx=(nx+1)//coarsen+1
    if (ny+1)%coarsen==0:
        coarsen_ny=(ny+1)//coarsen
    else:
        coarsen_ny=(ny+1)//coarsen+1
    if (nz+1)%coarsen==0:
        coarsen_nz=(nz+1)//coarsen
    else:
        coarsen_nz=(nz+1)//coarsen+1

    # create coarsened simulation box with one node per coarsened particle
    coarsen_sim_sld=np.zeros(coarsen_nx*coarsen_ny*coarsen_nz)
    num_prtcl=len(coarsen_sim_sld)

    # calculate rate to reactants and reagents
    # useful to take chemical progression into account
    chem_ratio_reac=mr_reac_arr*(1-chem_percentage)
    chem_ratio_reag=mr_reag_arr*chem_percentage

    # molar arte and slds of all compounds
    # at a particular chemical progression state
    # includes reactant and reagent  
    cur_mrs=np.concatenate((chem_ratio_reac, chem_ratio_reag))
    cur_slds=np.concatenate((sld_reac_arr, sld_reag_arr))


    # calculate num of occurances fro each compound
    num_comp=np.zeros(len(cur_mrs))
    for comp_idx, _ in enumerate(cur_mrs):
        norm_ratio=cur_mrs[comp_idx]/np.sum(cur_mrs)
        cur_num_comp=norm_ratio*num_prtcl
        num_comp[comp_idx]=np.floor(cur_num_comp)

    # manage nodes unassigned due to flooring above 
    remain_node=int(num_prtcl-np.sum(num_comp))
    for j in range(remain_node):
        if num_comp[j]!=0:
            num_comp[j]=num_comp[j]+1

    # assign slds to coarsened simulation box
    for comp_idx, _ in enumerate(num_comp):
        idx_start=int(np.sum(num_comp[0:comp_idx]))
        idx_end=int(np.sum(num_comp[0:comp_idx])+num_comp[comp_idx])
        coarsen_sim_sld[idx_start:idx_end]=cur_slds[comp_idx]
    np.random.shuffle(coarsen_sim_sld)   
    coarsen_sim_sld_3d=coarsen_sim_sld.reshape(coarsen_nx,coarsen_ny,coarsen_nz)

    # unpack coarsen
    # each particle is repeated ${coarsen} number of times
    sim_sld_3d=coarsen_sim_sld_3d
    for dim in range(3):
        sim_sld_3d=np.repeat(sim_sld_3d, coarsen, axis=dim)
    # cut at boundary of actual simulation box
    sim_sld_3d_cut=sim_sld_3d[0:nx+1, 0:ny+1, 0:nz+1]
    # 3d to 1d
    sim_sld=sim_sld_3d_cut.flatten()
    return sim_sld
