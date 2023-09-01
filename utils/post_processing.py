import os 
import cv2
import numpy as np 
from tqdm import tqdm 



def change_images_order(imgs_dir):
    imgs = os.listdir(imgs_dir)
    for img_name in tqdm(imgs):
        img = cv2.imread(os.path.join(imgs_dir,img_name))
        img_tmp = np.zeros_like(img)
        
        img_tmp[:,:128,:] = img[:,:128,:]
        img_tmp[:,128:3*128,:] = img[:,2*128:,:]
        img_tmp[:,3*128:,:] = img[:,128:2*128,:]
        
        cv2.imwrite(os.path.join(imgs_dir,img_name),img_tmp)
        
def get_only_view64(imgs_dir):
    imgs = [idx_name for idx_name in os.listdir(imgs_dir) if os.path.isdir(os.path.join(imgs_dir,idx_name))]
    #print(imgs)
    for img_name in tqdm(imgs):
        src_view64 = cv2.imread(os.path.join(imgs_dir,img_name,'rgb','000064.png'))
        #print(src_view64.shape)
        cv2.imwrite(f'/data/SHR-NeRF/logs/src64_test/{img_name}.png',src_view64)
        
if __name__ == '__main__':
    imgs_dir = '/data/srn_cars/cars_test' 
    get_only_view64(imgs_dir)
    