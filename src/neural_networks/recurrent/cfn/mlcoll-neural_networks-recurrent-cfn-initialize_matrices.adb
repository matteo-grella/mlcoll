------------------------------------------------------------------------------
--                               M L C O L L
--  M a c h i n e   L e a r n i n g   C o m p o n e n t   C o l l e c t i o n
--
--        Copyright 2009-2016 M. Grella, S. Cangialosi, E. Brambilla
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

separate (MLColl.Neural_Networks.Recurrent.CFN)

procedure Initialize_Matrices
  (Model : in out Model_Type) is
          
    Weights_Range : Real renames Model.Configuration.Random_Weights_Range;
begin

    if Model.Wc_In /= null 
      or else Model.Wf_In /= null 
      or else Model.Wi_In /= null 
      or else Model.Wc_In /= null 
      or else Model.Wf_Rec /= null 
      or else Model.Wi_Rec /= null 
      or else Model.Bf_In /= null 
      or else Model.Bi_In /= null  
      or else Model.W_Out /= null
      or else Model.B_Out /= null then
             
        raise CFN_Exception
          with "Matrices already initialized";
    end if;
      
    ----
    -- Input Weight Layer        
    ----
        
    Model.Wc_In := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Input_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Model.Wc_In.all := (others => (others => Get_Random_Weight (Weights_Range)));
      
    Model.Wf_In := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Input_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Model.Wf_In.all := (others => (others => Get_Random_Weight (Weights_Range)));
      
    Model.Wi_In := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Input_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Model.Wi_In.all := (others => (others => Get_Random_Weight (Weights_Range)));

    ----
    -- Recurrent Weight Layer
    ----      
      
    Model.Wf_Rec := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Model.Wf_Rec.all := (others => (others => Get_Random_Weight (Weights_Range)));
      
    Model.Wi_Rec := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Model.Wi_Rec.all := (others => (others => Get_Random_Weight (Weights_Range)));

    ----
    -- Input Bias       
    ----
      
    Model.Bf_In := new Real_Array
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);

    Model.Bf_In.all := (others => 0.0);
      
    Model.Bi_In := new Real_Array
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);

    Model.Bi_In.all := (others => 0.0);

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
    -- ADAM
    ----
        
    if Model.Configuration.Learning_Rule = ADAM then
        Model.Wcm_In      := new Real_Matrix (Model.Wc_In'Range (1), Model.Wc_In'Range (2));
        Model.Wcv_In      := new Real_Matrix (Model.Wc_In'Range (1), Model.Wc_In'Range (2));
        Model.Wcm_In.all  := (others => (others => 0.0)); 
        Model.Wcv_In.all  := (others => (others => 0.0));
         
        Model.Wim_In      := new Real_Matrix (Model.Wi_In'Range (1), Model.Wi_In'Range (2));
        Model.Wiv_In      := new Real_Matrix (Model.Wi_In'Range (1), Model.Wi_In'Range (2));
        Model.Wim_In.all  := (others => (others => 0.0)); 
        Model.Wiv_In.all  := (others => (others => 0.0));
         
        Model.Wfm_In      := new Real_Matrix (Model.Wf_In'Range (1), Model.Wf_In'Range (2));
        Model.Wfv_In      := new Real_Matrix (Model.Wf_In'Range (1), Model.Wf_In'Range (2));
        Model.Wfm_In.all  := (others => (others => 0.0)); 
        Model.Wfv_In.all  := (others => (others => 0.0));
         
        Model.Wim_Rec     := new Real_Matrix (Model.Wi_Rec'Range (1), Model.Wi_Rec'Range (2));
        Model.Wiv_Rec     := new Real_Matrix (Model.Wi_Rec'Range (1), Model.Wi_Rec'Range (2));
        Model.Wim_Rec.all := (others => (others => 0.0)); 
        Model.Wiv_Rec.all := (others => (others => 0.0));
         
        Model.Wfm_Rec     := new Real_Matrix (Model.Wf_Rec'Range (1), Model.Wf_Rec'Range (2));
        Model.Wfv_Rec     := new Real_Matrix (Model.Wf_Rec'Range (1), Model.Wf_Rec'Range (2));
        Model.Wfm_Rec.all := (others => (others => 0.0)); 
        Model.Wfv_Rec.all := (others => (others => 0.0));
         
         
        Model.Bim_In       := new Real_Array (Model.Bi_In'Range);
        Model.Biv_In       := new Real_Array (Model.Bi_In'Range);
        Model.Bim_In.all   := (others => 0.0);
        Model.Biv_In.all   := (others => 0.0);
         
        Model.Bfm_In       := new Real_Array (Model.Bf_In'Range);
        Model.Bfv_In       := new Real_Array (Model.Bf_In'Range);
        Model.Bfm_In.all   := (others => 0.0);
        Model.Bfv_In.all   := (others => 0.0);
        
        Model.Wm_Out     := new Real_Matrix (Model.W_Out'Range (1), Model.W_Out'Range (2));
        Model.Wv_Out     := new Real_Matrix (Model.W_Out'Range (1), Model.W_Out'Range (2));
        Model.Wm_Out.all := (others => (others => 0.0)); 
        Model.Wv_Out.all := (others => (others => 0.0));
            
        Model.Bm_Out     := new Real_Array (Model.B_Out'Range);
        Model.Bv_Out     := new Real_Array (Model.B_Out'Range);
        Model.Bm_Out.all := (others => 0.0);
        Model.Bv_Out.all := (others => 0.0);
    end if;
        
end Initialize_Matrices;
