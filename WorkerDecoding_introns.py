def run_single_fov_decoding(save_folder,fov,set_='',lib_fl = r'C:\Users\cfg001\Desktop\WTC11\Adam_introns_1_27_2024_withBlank.csv',redo=False):
    dec = decoder_simple(save_folder,fov,set_=set_)
    if not os.path.exists(dec.decoded_fl) or redo:
        dec.get_XH(dec.fov,set_,ncols=3,nbits=12,th_h=5000,filter_tag = '')#number of colors match 
        dec.XH = dec.XH[dec.XH[:,-4]>0.25] ### keep the spots that are correlated with the expected PSF for 60X
        dec.load_library(lib_fl,nblanks=-1)
        dec.ncols = 3
        get_intersV2(dec,nmin_bits=3,dinstance_th=2,enforce_color=True,enforce_set=17*3,redo=False)
        get_icodesV3(dec,nmin_bits=3,iH=-3) ### saves a decoded....npz
        print("Decoding completed for {}.".format(dec.fov))

import sys
sys.path.append(r'C:\Users\cfg001\Desktop\WTC11\NMERFISH')
from ioMicro import *


# load in all files from specified experiment
all_flds = glob.glob(r'Z:\Adam\E201_WTC11_WTday15__10_20_2023\H*_RMER_I*')
all_flds = np.array(all_flds)[np.argsort([get_iH(fld) for fld in all_flds])]
all_flds
print("Folders found:",len(all_flds))
fov = 'Conv_zscan__200.zarr'
save_folder = r'Z:\Adam\E201_WTC11_WTday15__10_20_2023\AnalysisDeconvolve_CG\RNA_intron'

def fov_to_dapi_features(fov='Conv_zscan__200',save_folder_ref=r'Z:\Adam\E201_WTC11_WTday15__10_20_2023\AnalysisDeconvolve_CG',
                         save_folder=r'Z:\Adam\E201_WTC11_WTday15__10_20_2023\AnalysisDeconvolve_CG\RNA_intron',tag_new='H1_RMER_I1',tag_ref = 'H1_DMER_1'):
    fov_ = fov.replace('.zarr','')
    fl = save_folder+os.sep+fov_+'--'+tag_new+'--dapiFeatures.npz'
    fl_ref = save_folder_ref+os.sep+fov_+'--'+tag_ref+'--dapiFeatures.npz'
    return fl,fl_ref
def load_segmentation(dec,segmentation_folder = r'Z:\Adam\E201_WTC11_WTday15__10_20_2023\AnalysisDeconvolve_CG\segmentationDAPI',segm_tag='H1_DMER_1'):
    fl_segm = segmentation_folder+os.sep+dec.fov.replace('.zarr','')+'--'+segm_tag+'--CYTO_segm.npz'
    segm,shape = np.load(fl_segm)['segm'],np.load(fl_segm)['shape']
    #segm_ = resize(segm,shape)
    dec.im_segm_ = segm
    dec.shape=shape
    dec.segm_tag=segm_tag
def get_dic_drift(dec):
    drifts,flds,fov_,fl_ref = np.load(dec.drift_fl,allow_pickle=True)
    return {os.path.basename(fld):drft[0] for drft,fld in zip(drifts,flds)}
def main_f_fov(save_folder =r'Z:\Adam\E201_WTC11_WTday15__10_20_2023\AnalysisDeconvolve_CG\RNA_intron',fov='Conv_zscan__050',set_ = '',ncols=3,
          scores_ref_fl=r'Z:\Adam\E201_WTC11_WTday15__10_20_2023\AnalysisDeconvolve_CG\RNA_intron\scoresRef.pkl',th=-0.75,force=False,segm_tag='H1_DMER_1',tag_new='H1_RMER_I1'):
    save_fld_cell = os.path.dirname(save_folder)+os.sep+'best_AdamIntrons'
    if not os.path.exists(save_fld_cell): os.makedirs(save_fld_cell)
    save_fl = save_fld_cell+os.sep+fov+'__XHfs_finedrft.npz'
    if not os.path.exists(save_fl) or force:
        dec = decoder_simple(save_folder,fov,set_)
        print(dec.decoded_fl)
        dec.ncols = ncols
        dec.load_decoded()
        print(dec.decoded_fl)
        dec.save_fl=save_fl
        #apply_fine_drift(dec,plt_val=True)
        #dec.save_folder= r'C:\Users\cfg001\Desktop\WTC11\flat_field'
        #apply_flat_field(dec,tag='Scope4_med_col_raw')
        #dec.save_folder= save_folder
        #scoresRefT = get_score_per_color(dec)
        scoresRefT = pickle.load(open(scores_ref_fl,'rb'))
        dec.dist_best = np.load(dec.decoded_fl)['dist_best']
        get_score_withRef(dec,scoresRefT,plt_val=True,gene=None,iSs = None,th_min=-7.5,include_dbits=True)
        dec.th=th
        plot_statistics(dec)
        
        #threshold the combined EM score
        keep = dec.scoreA>dec.th
        dec.XH_prunedf,dec.icodesNf=dec.XH_pruned[keep],dec.icodesN[keep]
        nbits = dec.XH_prunedf.shape[1]
        dec.XH_prunedF = np.concatenate([dec.XH_prunedf,np.repeat(dec.icodesNf,nbits).reshape(-1,nbits)[:,:,np.newaxis]],axis=-1)
        ### get the drift - to correct to the segmentation space
        dec.dic_drift = get_dic_drift(dec)
        
        load_segmentation(dec)
        #dec.im_segm_ = expand_segmentation(dec.im_segm_, nexpand=5)
        
        ### Compute drift between the segmentation file and the reference drift file
        fl,fl_ref = fov_to_dapi_features(tag_new=tag_new,tag_ref=segm_tag)
        txyz_segm = get_best_translation_points_dapiFeat(fl,fl_ref,resc=5,th=5)[0]
        tzxy_seg = -txyz_segm#dec.dic_drift[dec.segm_tag] #np.round([dic_drift[key][0] for key in dic_drift if cp.segm_tag in key]).astype(int)
        ### Augment the fitting data with cell id
        resc = dec.im_segm_.shape/dec.shape
        XH_ = dec.XH_prunedF.copy()
        XH_[:,:,:3] = XH_[:,:,:3]-tzxy_seg ### bring fits to cell segmentation space - modified to -
        XC = (np.nanmean(XH_[:,:,:3],axis=1)*resc).astype(int) #rescale to segmentation size
        dec.XC = XC
        keep = np.all(XC>=0,axis=-1)&np.all(XC<dec.im_segm_.shape,axis=-1)
        icells = np.zeros(len(XC))
        icells[keep]=dec.im_segm_[tuple(XC[keep].T)]
        nbits = XH_.shape[1]
        icells = np.repeat(icells,nbits).reshape(-1,nbits)[:,:,np.newaxis]
        XH_f = np.concatenate([XH_,icells],axis=-1)
        dec.XH_f=XH_f
        XH_fs=keep_best_per_cell_fast(XH_f,nbest=5)
        cell_ids,vols = np.unique(dec.im_segm_,return_counts=True)
        np.savez(dec.save_fl,XH_fs=XH_f)
        return dec



def main_f(fov,try_mode=False):
    set_=''
    
    
    if True:#not os.path.exists(fl):
        if try_mode:
            try:
                compute_drift_V2(save_folder,fov,all_flds,set_='',redo=False,gpu=True)
                run_single_fov_decoding(save_folder,fov)
                dec = main_f_fov(fov=fov,th=-0.75,force=False)
            except:
                print("Failed within the main analysis:",fov)
        else:
            compute_drift_V2(save_folder,fov,all_flds,set_='',redo=False,gpu=True)
            run_single_fov_decoding(save_folder,fov)
            dec = main_f_fov(fov=fov,th=-0.75,force=False)
    
    
    return fov
    
from multiprocessing import Pool, TimeoutError
    
if __name__ == '__main__':
    # start 4 worker processes
    fovs = [os.path.basename(fl) for fl in np.sort(glob.glob(all_flds[0]+os.sep+'*.zarr'))]
    fovs = [ fov.split(".zarr")[0] for fov in fovs ]
    #item = fovs[56]
    #main_f(item,try_mode=False)
    if True:
        with Pool(processes=4) as pool:
            print('starting pool')
            result = pool.map(main_f, fovs)
#conda activate cellpose2&&python D:\Carlos\NMERFISH\WorkerDecodingD111_introns.py
