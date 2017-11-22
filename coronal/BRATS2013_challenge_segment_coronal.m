
size_voxel1=[33 1 33];
half_size_voxel1=(size_voxel1-1)/2;
size_voxel2=[65 1 65];
half_size_voxel2=(size_voxel2-1)/2;
H0=240;
W0=240;
H1=H0+32;
W1=W0+32;
H2=H0+64;
W2=W0+64;

caffe_path = '/data1/NLPRMNT/zhaoxiaomei/crfasrnn-master/caffe-crfrnn';

model_def_file = 'deploy.prototxt';
model_file1 = 'patch_65_33_crf_coronal_iter_60000.caffemodel';

use_gpu = 1; 

addpath('/data1/NLPRMNT/zhaoxiaomei/matlab_tools/NIfTI_20140122');
addpath('/data1/NLPRMNT/zhaoxiaomei/matlab_tools/read_and_write_mha');
addpath(fullfile(caffe_path, 'matlab/caffe'));

caffe('reset');
caffe('set_device', 2); 
tvg_matcaffe_init(use_gpu, model_def_file, model_file1);

for n=301

    fprintf('processing %dth data...\n',n);
    save_index=17572+(n-301)*4;
    V_Flair_ns_address='../BRATS2013_Challenge_data_301/MR_Flair/MR_Flair_N4_ns.mha';
    V_T1c_ns_address='../BRATS2013_Challenge_data_301/MR_T1c/MR_T1c_N4_ns.mha';
    V_T2_ns_address='../BRATS2013_Challenge_data_301/MR_T2/MR_T2_N4_ns.mha';

    V_Flair_ns=mha_read_volume(V_Flair_ns_address);
    V_T1c_ns=mha_read_volume(V_T1c_ns_address);
    V_T2_ns=mha_read_volume(V_T2_ns_address);

    size_V=size(V_Flair_ns);
    result=zeros(size_V(1),size_V(2),size_V(3));
    %---------------------------------------------------
    V_Flair2_ns=zeros(size_V(1)+size_voxel2(1)-1,size_V(2),size_V(3)+size_voxel2(3)-1);
    V_T1c2_ns=zeros(size_V(1)+size_voxel2(1)-1,size_V(2),size_V(3)+size_voxel2(3)-1);
    V_T22_ns=zeros(size_V(1)+size_voxel2(1)-1,size_V(2),size_V(3)+size_voxel2(3)-1);
    V_Flair2_ns(half_size_voxel2(1)+1:half_size_voxel2(1)+size_V(1),:,half_size_voxel2(3)+1:half_size_voxel2(3)+size_V(3))=V_Flair_ns;
    V_T1c2_ns(half_size_voxel2(1)+1:half_size_voxel2(1)+size_V(1),:,half_size_voxel2(3)+1:half_size_voxel2(3)+size_V(3))=V_T1c_ns;
    V_T22_ns(half_size_voxel2(1)+1:half_size_voxel2(1)+size_V(1),:,half_size_voxel2(3)+1:half_size_voxel2(3)+size_V(3))=V_T2_ns;
     size_V2=size(V_Flair2_ns);
    %---------------------------------------------------
    for y=1:size_V(2)
        cur_time = fix(clock); 
		str = sprintf(' %.2d:%.2d:%.2d-----', cur_time(4), cur_time(5), cur_time(6)); 
        fprintf(str);		
        fprintf('processing %d - %dth ...%d\n',n,y,a);
		
		flair_y=V_Flair2_ns(:,y,:);
		area_non_0=sum(flair_y(:)>0);
		if area_non_0<=10
		    continue;
		end
		
        K=floor(size_V(1)/H0)+1;
        stride_H=H0;
        for k=1:K
            flair0=zeros(H0,size_voxel1(2),W0);
            t1c0=zeros(H0,size_voxel1(2),W0);
            t20=zeros(H0,size_voxel1(2),W0);
            flair0(1:min(size_V2(1)-((k-1)*stride_H),H2)-64,:,1:min(W2,size_V2(3))-64)=V_Flair2_ns((k-1)*stride_H+33:min(size_V2(1),(k-1)*stride_H+H2)-32,y-half_size_voxel2(2):y+half_size_voxel2(2),33:min(W2,size_V2(3))-32);
            t1c0(1:min(size_V2(1)-((k-1)*stride_H),H2)-64,:,1:min(W2,size_V2(3))-64)=V_T1c2_ns((k-1)*stride_H+33:min(size_V2(1),(k-1)*stride_H+H2)-32,y-half_size_voxel2(2):y+half_size_voxel2(2),33:min(W2,size_V2(3))-32);
            t20(1:min(size_V2(1)-((k-1)*stride_H),H2)-64,:,1:min(W2,size_V2(3))-64)=V_T22_ns((k-1)*stride_H+33:min(size_V2(1),(k-1)*stride_H+H2)-32,y-half_size_voxel2(2):y+half_size_voxel2(2),33:min(W2,size_V2(3))-32);

            flair1=zeros(H1,size_voxel1(2),W1);
            t1c1=zeros(H1,size_voxel1(2),W1);
            t21=zeros(H1,size_voxel1(2),W1);
            flair1(1:min(size_V2(1)-((k-1)*stride_H),H2)-32,:,1:min(W2,size_V2(3))-32)=V_Flair2_ns((k-1)*stride_H+17:min(size_V2(1),(k-1)*stride_H+H2)-16,y-half_size_voxel2(2):y+half_size_voxel2(2),17:min(W2,size_V2(3))-16);
            t1c1(1:min(size_V2(1)-((k-1)*stride_H),H2)-32,:,1:min(W2,size_V2(3))-32)=V_T1c2_ns((k-1)*stride_H+17:min(size_V2(1),(k-1)*stride_H+H2)-16,y-half_size_voxel2(2):y+half_size_voxel2(2),17:min(W2,size_V2(3))-16);
            t21(1:min(size_V2(1)-((k-1)*stride_H),H2)-32,:,1:min(W2,size_V2(3))-32)=V_T22_ns((k-1)*stride_H+17:min(size_V2(1),(k-1)*stride_H+H2)-16,y-half_size_voxel2(2):y+half_size_voxel2(2),17:min(W2,size_V2(3))-16);

            flair2=zeros(H2,size_voxel2(2),W2);
            t1c2=zeros(H2,size_voxel2(2),W2);
            t22=zeros(H2,size_voxel2(2),W2);
            flair2(1:min(H2,size_V2(1)-(k-1)*stride_H),:,1:min(W2,size_V2(3)))=V_Flair2_ns((k-1)*stride_H+1:min(size_V2(1),(k-1)*stride_H+H2),y-half_size_voxel2(2):y+half_size_voxel2(2),1:min(W2,size_V2(3)));
            t1c2(1:min(H2,size_V2(1)-(k-1)*stride_H),:,1:min(W2,size_V2(3)))=V_T1c2_ns((k-1)*stride_H+1:min(size_V2(1),(k-1)*stride_H+H2),y-half_size_voxel2(2):y+half_size_voxel2(2),1:min(W2,size_V2(3)));
            t22(1:min(H2,size_V2(1)-(k-1)*stride_H),:,1:min(W2,size_V2(3)))=V_T22_ns((k-1)*stride_H+1:min(size_V2(1),(k-1)*stride_H+H2),y-half_size_voxel2(2):y+half_size_voxel2(2),1:min(W2,size_V2(3)));

			flair00=reshape(flair0,H0,W0);
			t1c00=reshape(t1c0,H0,W0);
			t200=reshape(t20,H0,W0);
			flair11=reshape(flair1,H1,W1);
			t1c11=reshape(t1c1,H1,W1);
			t211=reshape(t21,H1,W1);
			flair22=reshape(flair2,H2,W2);
			t1c22=reshape(t1c2,H2,W2);
			t222=reshape(t22,H2,W2);
			inputData = {single(flair00);single(t1c00);single(t200);single(flair11);single(t1c11);single(t211);single(flair22);single(t1c22);single(t222)};
            scores = caffe('forward',inputData);
            for x=1:min(H0,size_V(1)-(k-1)*stride_H)
                for z=1:min(size_V(3),W0)
                    [max_prob,max_prob_index]=max(scores{1}(x,z,:));
                    result(x+(k-1)*stride_H,y,z)=max_prob_index-1;
                end
            end
        end
    end

    V_out_address=['result_BRATS2013_Challenge/VSD.segment_result_befor_postprocess_',num2str(n),'.',num2str(save_index),'.mha'];
    V_in=mha_read_volume2(V_Flair_ns_address);
    V_out=V_in;
    V_out.metaData.ElementType='MET_SHORT';
    V_out.pixelData=int16(result);
    mha_write_volume2(V_out_address,V_out,1);

end


