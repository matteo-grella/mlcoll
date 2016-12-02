------------------------------------------------------------------------------
--                               M L C O L L
--  M a c h i n e   L e a r n i n g   C o m p o n e n t   C o l l e c t i o n
--
--        Copyright 2009-2013 M. Grella, S. Cangialosi, E. Brambilla
--
--  This is free software; you can redistribute it and/or modify it under
--  terms of the GNU General Public License as published by the Free Software
--  Foundation; either version 2, or (at your option) any later version.
--  This software is distributed in the hope that it will be useful, but WITH
--  OUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
--  or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
--  for more details. Free Software Foundation, 59 Temple Place - Suite
--  330, Boston, MA 02111-1307, USA.
--
--  As a special exception, if other files instantiate generics from this
--  unit, or you link this unit with other files to produce an executable,
--  this unit does not by itself cause the resulting executable to be
--  covered by the GNU General Public License. This exception does not
--  however invalidate any other reasons why the executable file might be
--  covered by the GNU Public License.
--
------------------------------------------------------------------------------

pragma License (Modified_GPL);

separate (MLColl.Neural_Networks.Recurrent.GRU_With_Attention)

procedure Initialize_Matrices
  (Model : in out Model_Type) is
          
    Weights_Range : Real renames Model.Configuration.Random_Weights_Range;
begin

    if Model.Wc_In /= null 
      or else Model.Wr_In /= null 
      or else Model.Wz_In /= null 
      or else Model.Wc_In /= null 
      or else Model.Wc_Rec /= null 
      or else Model.Wr_Rec /= null 
      or else Model.Wz_Rec /= null 
      or else Model.Bc_In /= null 
      or else Model.Br_In /= null 
      or else Model.Bz_In /= null 
       
      or else Model.Wa_In /= null 
      or else Model.Wa_Rec /= null 
      or else Model.Wb_In /= null 
      or else Model.Ba_In /= null 
      or else Model.A_In /= null 
        
      or else Model.W_Out /= null
      or else Model.B_Out /= null then
             
        raise GRU_Exception
          with "Matrices already initialized";
    end if;
      
    ----
    -- Input Weight Layer        
    ----
        
    Model.Wc_In := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Input_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Model.Wc_In.all := (others => (others => Get_Random_Weight (Weights_Range)));
      
    Model.Wr_In := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Input_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Model.Wr_In.all := (others => (others => Get_Random_Weight (Weights_Range)));
      
    Model.Wz_In := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Input_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Model.Wz_In.all := (others => (others => Get_Random_Weight (Weights_Range)));

    ----
    -- Recurrent Weight Layer
    ----
        
    Model.Wc_Rec := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);

    Model.Wc_Rec.all := (others => (others => Get_Random_Weight (Weights_Range)));
      
    Model.Wr_Rec := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Model.Wr_Rec.all := (others => (others => Get_Random_Weight (Weights_Range)));
      
    Model.Wz_Rec := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Model.Wz_Rec.all := (others => (others => Get_Random_Weight (Weights_Range)));

    ----
    -- Input Bias       
    ----
        
    Model.Bc_In := new Real_Array
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);

    Model.Bc_In.all := (others => 0.0);
      
    Model.Br_In := new Real_Array
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);

    Model.Br_In.all := (others => 0.0);
      
    Model.Bz_In := new Real_Array
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);

    Model.Bz_In.all := (others => 0.0);

    ----
    -- Output Weight Layer       
    ----
        
    Model.W_Out := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Output_Layer_Size) - 1);
        
    Model.W_Out.all := (others => (others => Get_Random_Weight (Weights_Range)));
            
    ----
    -- Output Bias       
    ----

    Model.B_Out := new Real_Array
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Output_Layer_Size) - 1);

    Model.B_Out.all := (others => 0.0);
    
    ----
    -- Attention Layer       
    ----
    
    Model.Wa_In := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Input_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Model.Wa_In.all := (others => (others => Get_Random_Weight (Weights_Range)));
    
    Model.Ba_In := new Real_Array
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);

    Model.Ba_In.all := (others => 0.0);
    
    Model.Wa_Rec := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Model.Wa_Rec.all := (others => (others => Get_Random_Weight (Weights_Range)));
    
    Model.Wb_In := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Input_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Model.Wb_In.all := (others => (others => Get_Random_Weight (Weights_Range)));
    
    Model.A_In := new Real_Array
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);

    --Model.A_In.all := (others => 0.0);
    Model.A_In.all := (others => (Get_Random_Weight (Weights_Range)));
    
    ----
    -- ADAM
    ----
        
    if Model.Configuration.Learning_Rule = ADAM then
        Model.Wcm_In      := new Real_Matrix (Model.Wc_In'Range (1), Model.Wc_In'Range (2));
        Model.Wcv_In      := new Real_Matrix (Model.Wc_In'Range (1), Model.Wc_In'Range (2));
        Model.Wcm_In.all  := (others => (others => 0.0)); 
        Model.Wcv_In.all  := (others => (others => 0.0));
         
        Model.Wzm_In      := new Real_Matrix (Model.Wz_In'Range (1), Model.Wz_In'Range (2));
        Model.Wzv_In      := new Real_Matrix (Model.Wz_In'Range (1), Model.Wz_In'Range (2));
        Model.Wzm_In.all  := (others => (others => 0.0)); 
        Model.Wzv_In.all  := (others => (others => 0.0));
         
        Model.Wrm_In      := new Real_Matrix (Model.Wr_In'Range (1), Model.Wr_In'Range (2));
        Model.Wrv_In      := new Real_Matrix (Model.Wr_In'Range (1), Model.Wr_In'Range (2));
        Model.Wrm_In.all  := (others => (others => 0.0)); 
        Model.Wrv_In.all  := (others => (others => 0.0));
               
        Model.Wcm_Rec     := new Real_Matrix (Model.Wc_Rec'Range (1), Model.Wc_Rec'Range (2));
        Model.Wcv_Rec     := new Real_Matrix (Model.Wc_Rec'Range (1), Model.Wc_Rec'Range (2));
        Model.Wcm_Rec.all := (others => (others => 0.0)); 
        Model.Wcv_Rec.all := (others => (others => 0.0));
         
        Model.Wzm_Rec     := new Real_Matrix (Model.Wz_Rec'Range (1), Model.Wz_Rec'Range (2));
        Model.Wzv_Rec     := new Real_Matrix (Model.Wz_Rec'Range (1), Model.Wz_Rec'Range (2));
        Model.Wzm_Rec.all := (others => (others => 0.0)); 
        Model.Wzv_Rec.all := (others => (others => 0.0));
         
        Model.Wrm_Rec     := new Real_Matrix (Model.Wz_Rec'Range (1), Model.Wr_Rec'Range (2));
        Model.Wrv_Rec     := new Real_Matrix (Model.Wz_Rec'Range (1), Model.Wr_Rec'Range (2));
        Model.Wrm_Rec.all := (others => (others => 0.0)); 
        Model.Wrv_Rec.all := (others => (others => 0.0));
         
        Model.Bcm_In       := new Real_Array (Model.Bc_In'Range);
        Model.Bcv_In       := new Real_Array (Model.Bc_In'Range);
        Model.Bcm_In.all   := (others => 0.0);
        Model.Bcv_In.all   := (others => 0.0);
         
        Model.Bzm_In       := new Real_Array (Model.Bz_In'Range);
        Model.Bzv_In       := new Real_Array (Model.Bz_In'Range);
        Model.Bzm_In.all   := (others => 0.0);
        Model.Bzv_In.all   := (others => 0.0);
         
        Model.Brm_In       := new Real_Array (Model.Br_In'Range);
        Model.Brv_In       := new Real_Array (Model.Br_In'Range);
        Model.Brm_In.all   := (others => 0.0);
        Model.Brv_In.all   := (others => 0.0);
        
        Model.Wm_Out     := new Real_Matrix (Model.W_Out'Range (1), Model.W_Out'Range (2));
        Model.Wv_Out     := new Real_Matrix (Model.W_Out'Range (1), Model.W_Out'Range (2));
        Model.Wm_Out.all := (others => (others => 0.0)); 
        Model.Wv_Out.all := (others => (others => 0.0));
            
        Model.Bm_Out     := new Real_Array (Model.B_Out'Range);
        Model.Bv_Out     := new Real_Array (Model.B_Out'Range);
        Model.Bm_Out.all := (others => 0.0);
        Model.Bv_Out.all := (others => 0.0);
        
        -- Attention Layer
        
        Model.Wam_In      := new Real_Matrix (Model.Wa_In'Range (1), Model.Wa_In'Range (2));
        Model.Wav_In      := new Real_Matrix (Model.Wa_In'Range (1), Model.Wa_In'Range (2));
        Model.Wam_In.all  := (others => (others => 0.0)); 
        Model.Wav_In.all  := (others => (others => 0.0));
        
        Model.Wbm_In      := new Real_Matrix (Model.Wb_In'Range (1), Model.Wb_In'Range (2));
        Model.Wbv_In      := new Real_Matrix (Model.Wb_In'Range (1), Model.Wb_In'Range (2));
        Model.Wbm_In.all  := (others => (others => 0.0)); 
        Model.Wbv_In.all  := (others => (others => 0.0));
        
        Model.Wam_Rec      := new Real_Matrix (Model.Wa_Rec'Range (1), Model.Wa_Rec'Range (2));
        Model.Wav_Rec      := new Real_Matrix (Model.Wa_Rec'Range (1), Model.Wa_Rec'Range (2));
        Model.Wam_Rec.all  := (others => (others => 0.0)); 
        Model.Wav_Rec.all  := (others => (others => 0.0));
        
        Model.Bam_In     := new Real_Array (Model.Ba_In'Range);
        Model.Bav_In     := new Real_Array (Model.Ba_In'Range);
        Model.Bam_In.all := (others => 0.0);
        Model.Bav_In.all := (others => 0.0);
        
        Model.Am_In     := new Real_Array (Model.A_In'Range);
        Model.Av_In     := new Real_Array (Model.A_In'Range);
        Model.Am_In.all := (others => 0.0);
        Model.Av_In.all := (others => 0.0);
        
    end if;
        
end Initialize_Matrices;
