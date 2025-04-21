#!/usr/bin/python

#
#
######### Please add documentation
### This is for ADAMs project, it will be running the coimbinatorial imaging, using the fitting/dapi-feats algorithm with PSFs and Flat Field Corrections sourced from Scope4 (Bogdans Microscope)
### Started on 6/26/2024 at 10:22 am
### python workerADAMRAD21_CHATDNAd14_comb.py
#################################################################
from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys,numpy as np
# change this: 
#C:\Users\cfg001\Desktop\WTC11
master_analysis_folder = r'C:\Users\cfg001\Desktop\WTC11'
sys.path.append(master_analysis_folder)
from ioMicroBogdanVerified import *
#standard is 4, its number of colors +1 
ncols = 4

#change
#r'C:\Users\cfg001\Desktop\WTC11\flat_field\Scope4_med_col_raw'
psf_file = r'C:\Users\cfg001\Desktop\WTC11\psfs\psf_750_E217_H1_MAP2_EEF2_DCX_Scope4.npy'

#W:\Adam\E217_WTC11_day14_Rad21new
save_folder = r'W:\Adam\E217_WTC11_day14_Rad21_DNA_MERFISH\AnalysisDeconvolve_CGBogdanV'
#change to whatever Bogdan gives you
#C:\Users\cfg001\Desktop\WTC11\flat_field
flat_field_fl = r'C:\Users\cfg001\Desktop\WTC11\flat_field\Scope4_med_col_raw'


def compute_drift_features(save_folder,fov,all_flds,set_,redo=False,gpu=True):
    fls = [fld+os.sep+fov for fld in all_flds]
    for fl in fls:
    	#important to update here, not on ioMicro for the flat field, psf, etc...
        get_dapi_features(fl,save_folder,set_,gpu=gpu,im_med_fl = flat_field_fl+r'3.npz',
                    psf_fl = psf_file)
def main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method,icol_flat):
    '''
    Inputs: save_folder -> this is on the NAS,where the output is saved
    
    '''
    im_ = read_im(fld+os.sep+fov)
    im__ = np.array(im_[icol],dtype=np.float32)
    
    if old_method:
        ### previous method without deconvolution
        im_n = norm_slice(im__,s=30)
        #Xh = get_local_max(im_n,500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,dbscan=True,
        #      return_centers=False,mins=None,sigmaZ=1,sigmaXY=1.5)
        Xh = get_local_maxfast_tensor(im_n,th_fit=500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5,gpu=False)
    else:
        
        fl_med = flat_field_fl+str(icol)+'.npz'

        im_med = np.array(np.load(fl_med)['im'],dtype=np.float32)
        im_med = cv2.blur(im_med,(20,20))
        im__ = im__/im_med*np.median(im_med)

        Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=True,
                                deconv={'method':'wiener','beta':0.0001},
                                delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
        
    np.savez_compressed(save_fl,Xh=Xh)
def compute_fits(save_folder,fov,all_flds,redo=False,ncols=ncols,
                psf_file = psf_file,try_mode=True,old_method=False,redefine_color=None):
    
    psf = np.load(psf_file)
    for ifld,fld in enumerate(tqdm(all_flds)):
        if redefine_color is not None:
            ncols = len(redefine_color[ifld])
        for icol in range(ncols-1):
            ### new method
            if redefine_color is None:
                icol_flat = icol
            else:
                #print("ifld is: "+str(ifld))
                #print("icol is: "+str(icol))
                icol_flat = redefine_color[ifld][icol]
            tag = os.path.basename(fld)
            save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+tag+'--col'+str(icol)+'__Xhfits.npz'
            if not os.path.exists(save_fl) or redo:
                if try_mode:
                    try:
                        main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method,redefine_color)
                    except:
                        print("Failed",fld,fov,icol)
                else:
                    main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method,redefine_color)
                    

def main_f(set_ifov,try_mode=True):
    '''
    CHANGE PATH HERE
    '''
    #20230122_R120PVTS32RDNA for the data used in grant
    #r'\\merfish6\merfish6v2\DNA_FISH\20230212_R120PVTS32RDNABDNF\Analysis'
     ### save folder
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    set__ = ''
   
    all_flds = []
    redefine_color = []
   
    
    
    #change to Adam's NAS, using H*DNA for the naming scheme
    #all_flds_ = glob.glob(r'/mnt/Ren-Imaging_NAS/Adam/E200_WTC11_WTday3__8_31_2023/H*DMER*'+set__) ################################# modify for 3-col small tad 
    #redefine_color += [[0,1,2,3]]*len(all_flds_) #### three signal color
    #all_flds+=all_flds_

    #toAdd: H*RMER, mrna (Q), I

    #W:\Adam\E217_WTC11_day14_Rad21new
    all_flds_ = glob.glob(r'W:\Adam\E217_WTC11_day14_Rad21_DNA_MERFISH\H*DMER*'+set__) ################################# modify for 3-col small tad 
    redefine_color += [[0,1,2,3]]*len(all_flds_) #### three signal color
    all_flds+=all_flds_

    
    
    
   
    set_,ifov = set_ifov
    all_flds = [fld.replace(set__,set_) for fld in all_flds]
    fovs_fl = save_folder+os.sep+'fovs__'+set_+'.npy'
    #print(fovs_fl)
    #print(all_flds)
    if not os.path.exists(fovs_fl):
        fls = glob.glob(all_flds[0]+os.sep+'*.zarr')
        fovs = [os.path.basename(fl) for fl in fls]
        np.save(fovs_fl,fovs)
    else:
        fovs = np.load(fovs_fl)
    if ifov<len(fovs):
        fov = fovs[ifov]
        
        print("Computing fitting on: "+str(fov))
        print(len(all_flds),all_flds)
        compute_fits(save_folder,fov,all_flds,redo=False,try_mode=try_mode,redefine_color=redefine_color)
        print("Computing drift on: "+str(fov))
        compute_drift_features(save_folder,fov,all_flds,set_,redo=False,gpu=True)
        #compute_drift(save_folder,fov,all_flds,set_,redo=False)
       
        #compute_decoding(save_folder,fov,set_)
        
    return set_ifov
if __name__ == '__main__':
    # start 4 worker processes
    items = [(set_,ifov)for set_ in ['']
                        for ifov in range(301)]
    #items.reverse()
    print(items)
    main_f(("",20),try_mode=False)
    if True:
        with Pool(processes=6) as pool:
            print('starting pool')
            result = pool.map(main_f,items)
            try:
                #for i in range(0,166):
                print("engaging with try mode...")
                result = pool.map(main_f,items)
            except:
                    print("failed at: " )

            
