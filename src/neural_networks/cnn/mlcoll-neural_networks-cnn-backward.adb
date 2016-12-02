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

separate (MLColl.Neural_Networks.CNN)

procedure Backward
  (Model                  : in out Model_Type;
   CNN_Structure          : access CNN_Structure_Type;
   Gradient               : in out Gradient_Type) is

    Input_Layers          : Real_Array_Access_Array renames CNN_Structure.Sequence_Input;
    Input_Gradients       : Real_Array_Access_Array renames CNN_Structure.Sequence_Input_Gradients;
    Hidden_Layers_Deriv   : Real_Array_Access_Array renames CNN_Structure.Sequence_Hidden_Derivative;
    Output_Layer          : Real_Array renames CNN_Structure.Output_Layer.all;
    Output_Layer_Argmax   : Index_Type_Array renames CNN_Structure.Output_Layer_Argmax.all;
    Output_Error          : Real_Array renames CNN_Structure.Output_Error.all;

begin

    ---
    -- Mini Batch gradients calculation
    ---

    for J in Output_Layer'Range loop
        declare
            Argmax : constant Index_Type := Index_Type (Output_Layer_Argmax (J));
            Hidden_Layer_Gradient : Real;
        begin
            Hidden_Layer_Gradient := Hidden_Layers_Deriv (Argmax) (J) * Output_Error (J);

            Gradient.B_In (J) := Gradient.B_In (J) + Hidden_Layer_Gradient;

            for I in Gradient.W_In'Range (1) loop
                Gradient.W_In (I, J) := Gradient.W_In (I, J) + (Hidden_Layer_Gradient * Input_Layers(Argmax)(I));
                Input_Gradients(Argmax) (I) := Input_Gradients(Argmax) (I) + (Hidden_Layer_Gradient * Model.W_In (I, J));
            end loop;
        end;
    end loop;

end Backward;
