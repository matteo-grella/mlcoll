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

separate (MLColl.Neural_Networks.Recurrent.RNN)

procedure Initialize_Matrices
  (Model : in out Model_Type) is
    
    Weights_Range : Real renames Model.Configuration.Random_Weights_Range; 
begin

    if Model.W_In /= null 
      or else Model.B_In /= null 
      or else Model.W_Out /= null
      or else Model.B_Out /= null
      or else Model.W_Rec /= null then
             
        raise Multilayer_Perceptron_Exception
          with "Matrices already initialized";
    end if;
        
    ----
    -- Input Weight Layer        
    ----
        
    Model.W_In := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Input_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);

    Model.W_In.all := (others => (others => Get_Random_Weight (Weights_Range)));

    ----
    -- Input Bias       
    ----
        
    Model.B_In := new Real_Array
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);

    Model.B_In.all := (others => 0.0);

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
    -- Recurrent Weight Layer
    ----
        
    Model.W_Rec := new Real_Matrix
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1,
       Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);

    Model.W_Rec.all := (others => (others => Get_Random_Weight (Weights_Range)));        
    
    ----
    -- ADAM Parameters
    ----
        
    if Model.Configuration.Learning_Rule = ADAM then
        Model.Wm_In      := new Real_Matrix (Model.W_In'Range (1), Model.W_In'Range (2));
        Model.Wv_In      := new Real_Matrix (Model.W_In'Range (1), Model.W_In'Range (2));
        Model.Wm_In.all  := (others => (others => 0.0)); 
        Model.Wv_In.all  := (others => (others => 0.0));
            
        Model.Bm_In       := new Real_Array (Model.B_In'Range);
        Model.Bv_In       := new Real_Array (Model.B_In'Range);
        Model.Bm_In.all   := (others => 0.0);
        Model.Bv_In.all   := (others => 0.0);

        Model.Wm_Rec     := new Real_Matrix (Model.W_Rec'Range (1), Model.W_Rec'Range (2));
        Model.Wv_Rec     := new Real_Matrix (Model.W_Rec'Range (1), Model.W_Rec'Range (2));
        Model.Wm_Rec.all := (others => (others => 0.0)); 
        Model.Wv_Rec.all := (others => (others => 0.0));    
            
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
