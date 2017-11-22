
for n=301
    save_index=17572+(n-301)*4;
    
    V_Flair_ns_address='axial/BRATS2013_Challenge_data_301/MR_Flair/MR_Flair_N4_ns.mha';
    V_T1c_ns_address='axial/BRATS2013_Challenge_data_301/MR_T1c/MR_T1c_N4_ns.mha';
    V_T2_ns_address='axial/BRATS2013_Challenge_data_301/MR_T2/MR_T2_N4_ns.mha';
    V_Flair_ns=mha_read_volume(V_Flair_ns_address);
    V_T1c_ns=mha_read_volume(V_T1c_ns_address);
    V_T2_ns=mha_read_volume(V_T2_ns_address);
    
    result_axial_address=['axial\result_BRATS2013_Challenge_post_G_post_G\VSD.segment_result_after_postprocess_',num2str(n),'.',num2str(save_index),'.mha'];
    result_coronal_address=['coronal\result_BRATS2013_Challenge_post_G_post_G\VSD.segment_result_after_postprocess_',num2str(n),'.',num2str(save_index),'.mha'];
    result_sagittal_address=['sagittal\result_BRATS2013_Challenge_post_G_post_G\VSD.segment_result_after_postprocess_',num2str(n),'.',num2str(save_index),'.mha'];
    result_axial=mha_read_volume(result_axial_address);
    result_coronal=mha_read_volume(result_coronal_address);
    result_sagittal=mha_read_volume(result_sagittal_address);

    size_v=size(result_axial);
    result=zeros(size_v(1),size_v(2),size_v(3));

    for x=1:size_v(1)
        for y=1:size_v(2)
            for z=1:size_v(3)
                if (result_axial(x,y,z)>0)+(result_coronal(x,y,z)>0)+(result_sagittal(x,y,z)>0)>2
                    result(x,y,z)=2;
                end
                if (result_axial(x,y,z)==1)+(result_coronal(x,y,z)==1)+(result_sagittal(x,y,z)==1)>2
                    result(x,y,z)=1;
                end
                if (result_axial(x,y,z)>2)+(result_coronal(x,y,z)>2)+(result_sagittal(x,y,z)>2)>2
                    result(x,y,z)=3;
                end
                if (result_axial(x,y,z)>=4)+(result_coronal(x,y,z)>=4)+(result_sagittal(x,y,z)>=4)>2
                    result(x,y,z)=4;
                end
            end
        end
    end
    
    %=====================================
    V_out_address= ['\VSD.segment_result_after_vote_',num2str(n),'.',num2str(save_index),'.mha'];
    V_in=mha_read_volume2(V_Flair_ns_address);
    V_out=V_in;
    %     V_out.metaData.CompressedData='True';
    V_out.metaData.ElementType='MET_SHORT';
    result(:)=min(result(:),4);
    V_out.pixelData=int16(result);
    mha_write_volume2(V_out_address,V_out,1);
    
%     for z0=10%30:10:130
%         figure;
%         subplot(2,3,1);imshow(V_Flair_ns(:,:,z0),[0 255]);title(['Flair',num2str(n),'-',num2str(z0)]);
%         subplot(2,3,2);imshow(V_T1c_ns(:,:,z0),[0 255]);title('T1c');
%         subplot(2,3,3);imshow(V_T2_ns(:,:,z0),[0 255]);title('T2');
%         subplot(2,3,4);imshow(result_axial(:,:,z0),[0 4]);title('axial');
%         subplot(2,3,5);imshow(result_coronal(:,:,z0),[0 4]);title('coronal');
%         subplot(2,3,6);imshow(result(:,:,z0),[0 4]);title('after vote');
%     end
end
