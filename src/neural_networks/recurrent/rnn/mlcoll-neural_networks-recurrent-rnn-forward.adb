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

procedure Forward
  (Model                   : in     Model_Type;
   RNN_Structure           : access RNN_Structure_Type) is
    
    Input_Layers            : Real_Array_Access_Array         renames RNN_Structure.Sequence_Input;
    Hidden_Layers           : RNN_Array_Of_Float_Vectors_Type renames RNN_Structure.Sequence_Hidden;
    Hidden_Layers_Deriv     : RNN_Array_Of_Float_Vectors_Type renames RNN_Structure.Sequence_Hidden_Derivative;
    Output_Layers           : Real_Array_Access_Array         renames RNN_Structure.Sequence_Output;
      
begin
        
    for T in Input_Layers'First .. Input_Layers'Last loop
        Addition (Hidden_Layers (T).all, Model.B_In.all);
         
        for I in Hidden_Layers (T)'Range loop
            if Hidden_Layers (Extended_Index_Type (T) - 1) (I) /= 0.0 then
                for J in Model.W_Rec'Range (2) loop
                    Hidden_Layers (T) (J) := Hidden_Layers (T) (J) 
                      + Hidden_Layers (Extended_Index_Type (T) - 1) (I) * Model.W_Rec (I, J);
                end loop;
            end if;
        end loop;
         
        for I in Input_Layers (T)'Range loop
            if Input_Layers (T) (I) /= 0.0 then -- optimization
                for J in Model.W_In'Range (2) loop
                    Hidden_Layers (T) (J) := Hidden_Layers (T) (J) + Input_Layers (T) (I) * Model.W_In (I, J);
                end loop;
            end if;
        end loop;
         
        Combined_Activation_Deriv
          (Activation_Function => Model.Configuration.Activation_Function_Name,
           V1                  => Hidden_Layers (T).all,
           V2                  => Hidden_Layers_Deriv (T).all);
              
        VM_Product
          (V_Out => Output_Layers (T).all, 
           V_In  => Hidden_Layers (T).all,
           M_In  => Model.W_Out.all);

        Addition (V1_Out => Output_Layers (T).all, V2 => Model.B_Out.all);
    end loop;
    
end Forward;
