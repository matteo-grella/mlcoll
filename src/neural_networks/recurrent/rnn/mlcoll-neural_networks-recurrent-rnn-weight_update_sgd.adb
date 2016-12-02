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

procedure Weight_Update_SGD
  (Model            : in out Model_Type;
   Gradient         : in     Gradient_Type) is

    LR    : Real renames Model.Learning_Rate;
begin

    -- Update Input Weight
    for I in Model.W_In'Range (1) loop
        if Gradient.Input_Non_Zero(I) then
            for J in Model.W_In'Range (2) loop
                Model.W_In (I, J) := Model.W_In (I, J) - (LR * Gradient.W_In (I, J));
            end loop;
        end if;
    end loop;

    -- Update Recurrent Weight
    for I in Model.W_Rec'Range (1) loop
        for J in Model.W_Rec'Range (2) loop
            Model.W_Rec (I, J) := Model.W_Rec (I, J) - (LR * Gradient.W_Rec (I, J));
        end loop;
    end loop;

    -- Update Input Bias
    for I in Model.B_In'Range loop
        Model.B_In (I) := Model.B_In (I) - (LR * Gradient.B_In (I));
    end loop;

    -- Update Output Weight
    for I in Model.W_Out'Range (1) loop
        for J in Model.W_Out'Range (2) loop
            Model.W_Out (I, J) := Model.W_Out (I, J) - (LR * Gradient.W_Out (I, J));
        end loop;
    end loop;

    -- Update Output Bias
    for I in Model.B_Out'Range loop
        Model.B_Out (I) := Model.B_Out (I) - (LR * Gradient.B_Out (I));
    end loop;

end Weight_Update_SGD;
