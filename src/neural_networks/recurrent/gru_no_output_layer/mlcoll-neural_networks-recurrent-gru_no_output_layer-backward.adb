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

separate (MLColl.Neural_Networks.Recurrent.GRU_No_Output_Layer)

procedure Backward
  (Model                     : in out Model_Type;
   GRU_Structure             : in out GRU_Structure_Type) is
        
    Gradient                  : Gradient_Type renames GRU_Structure.Gradient;
    Gradient_Input            : Real_Array_Access_Array renames GRU_Structure.Sequence_Input_Gradients;
        
    Input_Layers              : Real_Array_Access_Array renames GRU_Structure.Sequence_Input;
    Reset_Activations         : Real_Array_Access_Array renames GRU_Structure.Sequence_Reset_Activations;
    Interpolate_Activations   : Real_Array_Access_Array renames GRU_Structure.Sequence_Interpolate_Activations;
    Candidate_Activations     : Real_Array_Access_Array renames GRU_Structure.Sequence_Candidate_Activations;
    Hidden_Layers             : Real_Array_Access_Array renames GRU_Structure.Sequence_Hidden;
    --Output_Layers             : Real_Array_Access_Array renames GRU_Structure.Sequence_Output;
    Hidden_Error              : Real_Array_Access_Array renames GRU_Structure.Sequence_Hidden_Error;
        
    Delta_H :  Real_Array_Access := new Real_Array
      (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) - 1);
        
    Delta_R :  Real_Array_Access := new Real_Array (Delta_H.all'Range);
    Delta_Z :  Real_Array_Access := new Real_Array (Delta_H.all'Range);
    Delta_C :  Real_Array_Access := new Real_Array (Delta_H.all'Range);
begin
    
    Gradient.Count := Input_Layers'Length;
    
    for T in reverse Input_Layers'Range loop
        declare
            Dha :  Real_Array (Delta_H.all'Range);
            Dhb :  Real_Array (Dha'Range);
            Dhc :  Real_Array (Dha'Range);
            Dhd :  Real_Array (Dha'Range);
            Dc  :  Real_Array (Dha'Range);
        begin
            
            if T < Input_Layers'Last then
                Element_Product_Sub_One
                  (V1 => Dha,
                   V2 => Delta_H.all,
                   V3 => Interpolate_Activations (T + 1).all);
            end if;
            
            MV_Product
              (V_Out => dhb,
               M_In  => Model.Wr_Rec.all ,
               V_In  => Delta_R.all);
            
            MV_Product
              (V_Out => dhc,
               M_In  => Model.Wz_Rec.all ,
               V_In  => Delta_Z.all);
            
            if T < Input_Layers'Last then
                MV_Product
                  (V_Out => Dhd,
                   M_In  => Model.Wc_Rec.all ,
                   V_In  => Delta_C.all);
               
                Element_Product
                  (V1_Out => Dhd,
                   V2     => Reset_Activations (T + 1).all);   
            end if;

            for I in Delta_H.all'Range loop
                Delta_H (I) := Dha (I) + Dhb (I) + Dhc (I) + Dhd (I) + Hidden_Error (T)(I);
            end loop;
            
            for I in Delta_C'Range loop
                Delta_C (I) := (Delta_H (I) * Interpolate_Activations (T) (I)) * (1.0 - Candidate_Activations (T) (I) ** 2);
            end loop;
           
            MV_Product
              (V_Out => Delta_R.all,
               M_In  => Model.Wc_Rec.all ,
               V_In  => Delta_C.all);
          
            if T > Hidden_Layers'First then
                for I in Delta_R'Range loop
                    Delta_R (I) := (Delta_R (I) * Hidden_Layers (T - 1) (I)) * Reset_Activations (T) (I) * (1.0 - Reset_Activations (T) (I));
                end loop;
            
                for I in Delta_Z'Range loop
                    Delta_Z (I) := (Delta_H (I) * (Candidate_Activations (T) (I) - Hidden_Layers (T - 1) (I))) * Interpolate_Activations (T) (I) * (1.0 - Interpolate_Activations (T) (I));
                end loop;
            end if;
            
            VV_Add_Product
              (M_Out => Gradient.Wr_In,
               V1    => Input_Layers (T).all,
               V2    => Delta_R.all);
                
            VV_Add_Product 
              (M_Out => Gradient.Wz_In,
               V1    => Input_Layers (T).all, 
               V2    => Delta_Z.all);
                
            VV_Add_Product 
              (M_Out => Gradient.Wc_In, 
               V1    => Input_Layers (T).all, 
               V2    => Delta_C.all);
          
            Addition (V1_Out => Gradient.Br_In, V2 => Delta_R.all);
            Addition (V1_Out => Gradient.Bz_In, V2 => Delta_Z.all);
            Addition (V1_Out => Gradient.Bc_In, V2 => Delta_C.all);
         
            for I in Input_Layers (T)'Range loop
                for J in Gradient.Br_In'Range loop
                    Gradient_Input (T) (I) := Gradient_Input (T) (I) + Model.Wr_In (I, J) * Delta_R.all (J);
                    Gradient_Input (T) (I) := Gradient_Input (T) (I) + Model.Wz_In (I, J) * Delta_Z.all (J);
                    Gradient_Input (T) (I) := Gradient_Input (T) (I) + Model.Wc_In (I, J) * Delta_C.all (J);
                end loop;
            end loop;
          
            if T > Hidden_Layers'First then
                VV_Add_Product
                  (M_Out => Gradient.Wr_Rec,
                   V1    => Hidden_Layers (T - 1).all, 
                   V2    => Delta_R.all);
                
                VV_Add_Product
                  (M_Out => Gradient.Wz_Rec,
                   V1    => Hidden_Layers (T - 1).all, 
                   V2    => Delta_Z.all);
            
                Element_Product
                  (V1 => Dc,
                   V2 => Reset_Activations (T).all,
                   V3 => Hidden_Layers (T - 1).all);
            end if;
            
            VV_Add_Product
              (M_Out => Gradient.Wc_Rec, 
               V1    => Dc, 
               V2    => Delta_C.all);
        end;
    end loop;
    
    Free (Delta_H);
    Free (Delta_R);
    Free (Delta_Z);
    Free (Delta_C);
end Backward;
