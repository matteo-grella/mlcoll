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

procedure Backward
  (Model                  : in out Model_Type;
   RNN_Structure          : access RNN_Structure_Type) is
        
    Input_Layers          : Real_Array_Access_Array renames RNN_Structure.Sequence_Input;
    Hidden_Layers         : RNN_Array_Of_Float_Vectors_Type renames RNN_Structure.Sequence_Hidden;
    Hidden_Layers_Deriv   : RNN_Array_Of_Float_Vectors_Type renames RNN_Structure.Sequence_Hidden_Derivative;
    Output_Error          : Real_Array_Access_Array renames RNN_Structure.Sequence_Output_Error;
    Input_Gradients       : Real_Array_Access_Array renames RNN_Structure.Sequence_Input_Gradients;
      
    Gradient              : Gradient_Type renames RNN_Structure.Gradient;
    
    Delta_T :  Real_Array_Access := new Real_Array
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1); 
    --GB_In
      
    Delta_T2 : Real_Array_Access := new Real_Array
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
      
    Delta_T3 : Real_Array_Access := new Real_Array
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
       
    Embeddings_Learning_Rate : constant Real := 1.0; --0.1;
begin
             
    ---
    -- Mini Batch gradients calculation
    ---
        
    for T in reverse Input_Layers'Range loop

        VV_Add_Product 
          (M_Out => Gradient.W_Out,
           V1    => Hidden_Layers (T).all,
           V2    => Output_Error (T).all);
        
        Addition
          (V1_Out => Gradient.B_Out,
           V2     => Output_Error (T).all);
               
        MV_Product (V_Out => Delta_T2.all,
                    M_In  => Model.W_Out.all,
                    V_In  => Output_Error (T).all);
         
        Delta_T.all := (others => 0.0);
            
        Addition
          (V1_Out => Delta_T.all, 
           V2     => Delta_T2.all);
         
        Addition
          (V1_Out => Delta_T.all, 
           V2     => Delta_T3.all);
        
        Element_Product 
          (V1_Out => Delta_T.all,
           V2     => Hidden_Layers_Deriv (T).all);
         
        Addition
          (V1_Out => Gradient.B_In, 
           V2     => Delta_T.all);
            
        VV_Add_Product 
          (M_Out => Gradient.W_Rec,
           V1    => Hidden_Layers (T - 1).all,
           V2    => Delta_T.all);
              
        VV_Add_Product
          (M_Out => Gradient.W_In,
           V1    => Input_Layers (T).all,
           V2    => Delta_T.all);
              
        -- Accumulate Input Gradients
            
        for I in Input_Layers (T)'Range loop
            for J in Gradient.B_In'Range loop
                Input_Gradients (T) (I) := Input_Gradients (T) (I) + Model.W_In (I, J) * Delta_T (J);
            end loop;
            
            Input_Gradients (T) (I) := Input_Gradients (T) (I) * Embeddings_Learning_Rate;
        end loop;
        
        MV_Product
          (V_Out => Delta_T3.all,
           M_In  => Model.W_Rec.all,
           V_In  => Delta_T.all);
    end loop;
      
    Free (Delta_T);
    Free (Delta_T2);
    Free (Delta_T3);

end Backward;

