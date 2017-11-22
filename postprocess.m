
%-------------------------------------------------------
%flags
flag_remove_super_high_intensity=1;
flag_remove_below_3th=1;
flag_remove_small_area=1;
flag_imfill=1;
flag_correct_necrosis=1;
flag_change_edema_to_nonenhancing=1;
%-------------------------------------------------------
for n=301
    fprintf('processing %dth data...\n',n);
    %----------------------------------------------------
    %read mha
    save_index=17572+(n-301)*4;
    V_Flair_ns_address='axial/BRATS2013_Challenge_data_301/MR_Flair/MR_Flair_N4_ns.mha';
    V_T1c_ns_address='axial/BRATS2013_Challenge_data_301/MR_T1c/MR_T1c_N4_ns.mha';
    V_T2_ns_address='axial/BRATS2013_Challenge_data_301/MR_T2/MR_T2_N4_ns.mha';
    V_result_before_address=['axial\result_BRATS2013_Challenge_post_G\VSD.segment_result_before_postprocess_',num2str(n),'.',num2str(save_index),'.mha'];
     %=====================================
    V_out_address= ['axial\result_BRATS2013_Challenge_post_G_post_G\VSD.segment_result_after_postprocess_',num2str(n),'.',num2str(save_index),'.mha'];
    th5=100;
    V_Flair_ns=mha_read_volume(V_Flair_ns_address);
    V_T1c_ns=mha_read_volume(V_T1c_ns_address);
    V_T2_ns=mha_read_volume(V_T2_ns_address);
    V_result_before=mha_read_volume(V_result_before_address);
    size_V=size(V_Flair_ns);
    result=int16(V_result_before);
    %----------------------------------------------------
    %     remove super high intensity areas
    if flag_remove_super_high_intensity>0
        result_bw=~(~abs(result));
        L=bwlabeln(result_bw);
        num=max(max(max(L)));
        flair_v_mean=zeros(1,num);
        t2_v_mean=zeros(1,num);
        for i=1:num%caculate volume
            mask=int16(~abs(L-i));
            flair_v=V_Flair_ns.*mask;
            t2_v=V_T2_ns.*mask;
            flair_v_mean(i)=sum(flair_v(:))/sum(mask(:));
            t2_v_mean(i)=sum(t2_v(:))/sum(mask(:));
        end
        remove_flag=zeros(1,num);
        for i=1:num
            if flair_v_mean(i)>150&&t2_v_mean(i)>150
                remove_flag(i)=1;
            end
        end
       for z=1:size_V(3)
           for x=1:size_V(1)
               for y=1:size_V(2)
                   if L(x,y,z)~=0&&remove_flag(L(x,y,z))==1
                       result(x,y,z)=0;
                   end
               end
           end
       end
    end
    %====================================
    %=====================================
    result(:)=min(result(:),4);
    r_necrosis=~abs(result-1);
    r_edema=~abs(result-2);
    r_non_enhancing=~abs(result-3);
    r_enhancing=~abs(result-4);
    
    V_Flair_ns=single(uint8(V_Flair_ns));
    r=abs(r_edema+r_necrosis+r_non_enhancing+r_enhancing);
    V_flair_whole=V_Flair_ns.*r;
    flair_mean=sum(V_flair_whole(:))/sum(r(:));
    
    V_T2_ns=single(uint8(V_T2_ns));
    V_t2_whole=V_T2_ns.*r;
    t2_mean=sum(V_t2_whole(:))/sum(r(:));
    %=====================================
    %----------------------------------------------------
    if flag_remove_below_3th>0
        for z=1:size_V(3)
           for x=1:size_V(1)
               for y=1:size_V(2)
                   if (V_Flair_ns(x,y,z)<0.8*flair_mean&&V_T1c_ns(x,y,z)<125&&V_T2_ns(x,y,z)<0.9*t2_mean&&result(x,y,z)<4)
                       result(x,y,z)=0;
                   end
               end
           end
       end
    end
    %----------------------------------------------------
    %remove small area
    if flag_remove_small_area>0
        result_bw=~(~abs(result));
        L=bwlabeln(result_bw);
        num=max(max(max(L)));
        volume=zeros(1,num);
        for i=1:num%caculate volume
            volume(i)=sum(L(:)==i);
        end
        remove_flag=zeros(1,num);
        volume_max=max(volume);
        for i=1:num
            if volume(i)/volume_max<0.1  
                remove_flag(i)=1;
            end
        end
       for z=1:size_V(3)
           for x=1:size_V(1)
               for y=1:size_V(2)
                   if L(x,y,z)~=0&&remove_flag(L(x,y,z))==1
                       result(x,y,z)=0;
                   end
               end
           end
       end
    end
    %----------------------------------------------------
    %remove small area--enhancing
    if flag_remove_small_area>0
        result_bw=~abs(result-4);
        L=bwlabeln(result_bw);
        num=max(max(max(L)));
        volume=zeros(1,num);
        for i=1:num%caculate volume
            volume(i)=sum(L(:)==i);
        end
        remove_flag=zeros(1,num);
        volume_max=max(volume);
        for i=1:num
            if volume(i)/volume_max<0.1  
                remove_flag(i)=1;
            end
        end
       for z=1:size_V(3)
           for x=1:size_V(1)
               for y=1:size_V(2)
                   if L(x,y,z)~=0&&remove_flag(L(x,y,z))==1
                       result(x,y,z)=2;
                   end
               end
           end
       end
    end
    %----------------------------------------------------
    %----------------------------------------------------
    %fill holes
    if flag_imfill>0
        for z=1:size_V(3)
            I=mat2gray(result(:,:,z));
            sum_I_true=sum(I(:)>0);
            I_bw=im2bw(I,0.1);
            if sum_I_true>0
                I_bw2=imfill(I_bw,'holes');
                result(:,:,z)=result(:,:,z)+int16(I_bw2-I_bw);
            end
        end
    end
    %----------------------------------------------------
    %----------------------------------------------------
    %fill enhancing holes
    %if flag_imfill>0
    %    for z=1:size_V(3)
    %        I_bw=~(result(:,:,z)-4);
    %        I_bw2=imfill(I_bw,'holes');
    %        for x=1:size_V(1)
    %            for y=1:size_V(2)
    %                if I_bw2(x,y)-I_bw(x,y)>0
    %                    result(x,y,z)=1;
    %                end
    %            end
    %        end
    %    end
    %end
    %-----------------------------------------------
    if flag_correct_necrosis>0
        for z=1:size_V(3)
            for x=1:size_V(1)
                for y=1:size_V(2)
                    if V_T1c_ns(x,y,z)<100&&result(x,y,z)>=4
                        result(x,y,z)=1;
                    end
                end
            end
        end
    end
    %=====================================
    result(:)=min(result(:),4);
    r_necrosis=~abs(result-1);
    %     r_edema=~abs(result-2);
    %     r_non_enhancing=~abs(result-3);
    r_enhancing=~abs(result-4);
    if sum(r_enhancing(:))==0
        result(x,y,z)=4;
    end
    r=r_enhancing+r_necrosis;
    fprintf('%d 增强核和坏死的体积 =%d, 增强核体积 = %d - %f, 坏死体积= %d - %f \n',n,sum(r(:)>0),sum(r_enhancing(:)),sum(r_enhancing(:))/sum(result(:)>0),sum(r_necrosis(:)),sum(r_necrosis(:))/sum(result(:)>0));
    result_b=result;
    %===========================================
    if flag_change_edema_to_nonenhancing>0
    if sum(r_enhancing(:))/sum(result(:)>0)<0.05
        for z=1:size_V(3)
            for x=1:size_V(1)
                for y=1:size_V(2)
                    if V_T1c_ns(x,y,z)<85&&result(x,y,z)==2
                        result(x,y,z)=3;
                    end
                end
            end
        end
    end
    end
    %=====================================
    %=====================================
    V_in=mha_read_volume2(V_Flair_ns_address);
    V_out=V_in;
    %     V_out.metaData.CompressedData='True';
    V_out.metaData.ElementType='MET_SHORT';
    result(:)=min(result(:),4);
    V_out.pixelData=int16(result);
    mha_write_volume2(V_out_address,V_out,1);
    
%      for z0=30:10:size_V(3)
%         figure;
%         subplot(2,3,1);imshow(V_Flair_ns(:,:,z0),[0 255]);title(['Flair',num2str(n),'-',num2str(z0)]);
%         subplot(2,3,2);imshow(V_T1c_ns(:,:,z0),[0 255]);title('T1c');
%         subplot(2,3,3);imshow(V_T2_ns(:,:,z0),[0 255]);title('T2');
% %         subplot(2,3,4);imshow(truth(:,:,z0),[0 4]);title('truth');
%         subplot(2,3,5);imshow(V_result_before(:,:,z0),[0 4]);title('before post-process');
%         subplot(2,3,6);imshow(result(:,:,z0),[0 4]);title('after post-process');
%     end
    

end