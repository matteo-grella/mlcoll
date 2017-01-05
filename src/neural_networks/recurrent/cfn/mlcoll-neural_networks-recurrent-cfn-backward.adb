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

procedure Backward
  (Model                     : in out Model_Type;
   CFN_Structure             : in out CFN_Structure_Type) is
        
    Gradient                  : Gradient_Type renames CFN_Structure.Gradient;
    Gradient_Input            : Real_Array_Access_Array renames CFN_Structure.Sequence_Input_Gradients;
        
    Input_Layers              : Real_Array_Access_Array renames CFN_Structure.Sequence_Input;
    Forget_Activations        : Real_Array_Access_Array renames CFN_Structure.Sequence_Forget_Activations;
    Input_Activations         : Real_Array_Access_Array renames CFN_Structure.Sequence_Input_Activations;
    Candidate_Activations     : Real_Array_Access_Array renames CFN_Structure.Sequence_Candidate_Activations;
    Hidden_Activations        : Real_Array_Access_Array renames CFN_Structure.Sequence_Hidden_Activations;
    Hidden_Layers             : RNN_Array_Of_Float_Vectors_Type renames CFN_Structure.Sequence_Hidden;
    Output_Error              : Real_Array_Access_Array renames CFN_Structure.Sequence_Output_Error;
        
    Delta_H :  Real_Array_Access := new Real_Array
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Delta_F :  Real_Array_Access := new Real_Array (Delta_H.all'Range);
    Delta_I :  Real_Array_Access := new Real_Array (Delta_H.all'Range);
    Delta_C :  Real_Array_Access := new Real_Array (Delta_H.all'Range);

    Embeddings_Learning_Rate : constant Real := 1.0;
begin
      
    for T in reverse Input_Layers'Range loop
        declare
            Dha :  Real_Array (Delta_H.all'Range);
            Dhb :  Real_Array (Dha'Range);
            Dhc :  Real_Array (Dha'Range);     
            Dhe :  Real_Array (Dha'Range);
            Ds  :  Real_Array (Dha'Range);
            
        begin 
            VV_Add_Product
              (M_Out => Gradient.W_Out,
               V1    => Hidden_Layers (T).all,
               V2    => Output_Error (T).all);
            
            Addition
              (V1_Out => Gradient.B_Out,
               V2     => Output_Error (T).all);

            if T < Input_Layers'Last then
                Element_Product (V1 => Ds,
                                 V2 => Delta_H.all,
                                 V3 => Forget_Activations (T + 1).all);
                
                for I in Dha'Range loop
                    Dha (I) := Ds(I) * (1.0 - Hidden_Activations(T + 1)(I) ** 2); -- tanh derivative
                end loop;                
            end if;
            
            MV_Product
              (V_Out => dhb,
               M_In  => Model.Wf_Rec.all ,
               V_In  => Delta_F.all);
            
            MV_Product
              (V_Out => dhc,
               M_In  => Model.Wi_Rec.all ,
               V_In  => Delta_I.all);
            
            MV_Product
              (V_Out => Dhe,
               M_In  => Model.W_Out.all ,
               V_In  => Output_Error (T).all);
            
            for I in Delta_H.all'Range loop
                Delta_H (I) := Dha (I) + Dhb (I) + Dhc (I) + Dhe (I);
            end loop;
            
         
            for I in Delta_C'Range loop
                Delta_C (I) := (Delta_H (I) * Input_Activations (T) (I)) * (1.0 - Candidate_Activations (T) (I) ** 2);
            end loop;
           
            for I in Delta_F'Range loop
                Delta_F (I) := (Delta_H (I) * Hidden_Activations (T) (I)) * Forget_Activations (T) (I) * (1.0 - Forget_Activations (T) (I));
            end loop;
            
            for I in Delta_I'Range loop
                Delta_I (I) := (Delta_H (I) * Candidate_Activations (T) (I)) * Input_Activations (T) (I) * (1.0 - Input_Activations (T) (I));
            end loop;
        
            VV_Add_Product
              (M_Out => Gradient.Wf_In,
               V1    => Input_Layers (T).all,
               V2    => Delta_F.all);
                
            VV_Add_Product
              (M_Out => Gradient.Wi_In,
               V1    => Input_Layers (T).all, 
               V2    => Delta_I.all);
                
            VV_Add_Product
              (M_Out => Gradient.Wc_In, 
               V1    => Input_Layers (T).all, 
               V2    => Delta_C.all);
          
            Addition (V1_Out => Gradient.Bf_In, V2 => Delta_F.all);
            Addition (V1_Out => Gradient.Bi_In, V2 => Delta_I.all);
            
            -- Accumulate Input Gradients
            for I in Input_Layers (T)'Range loop
                for J in Gradient.Bf_In'Range loop
                    Gradient_Input (T) (I) := Gradient_Input (T) (I) + Model.Wf_In (I, J) * Delta_F.all (J);
                    Gradient_Input (T) (I) := Gradient_Input (T) (I) + Model.Wi_In (I, J) * Delta_I.all (J);
                    Gradient_Input (T) (I) := Gradient_Input (T) (I) + Model.Wc_In (I, J) * Delta_C.all (J);
                end loop;
                        
                Gradient_Input (T) (I) := Gradient_Input (T) (I) * Embeddings_Learning_Rate;
            end loop;
          
            VV_Add_Product
              (M_Out => Gradient.Wf_Rec,
               V1    => Hidden_Layers (T - 1).all, 
               V2    => Delta_F.all);
                
            VV_Add_Product
              (M_Out => Gradient.Wi_Rec,
               V1    => Hidden_Layers (T - 1).all, 
               V2    => Delta_I.all);
        end;
    end loop;
    
    Free (Delta_H);
    Free (Delta_F);
    Free (Delta_I);
    Free (Delta_C);
    
end Backward;

