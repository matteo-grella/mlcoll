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

separate (MLColl.Neural_Networks.Recurrent.GRU)

procedure Weight_Update_SGD
  (Model      : in out Model_Type;
   Gradient   : in     Gradient_Type) is

    LR    : Real renames Model.Learning_Rate;
begin

    -- Update Input Weights

    for I in Model.Wr_In'Range (1) loop
        for J in Model.Wr_In'Range (2) loop
            Model.Wr_In (I, J) := Model.Wr_In (I, J) - (Gradient.Wr_In (I, J) * LR);
        end loop;
    end loop;

    for I in Model.Wz_In'Range (1) loop
        for J in Model.Wz_In'Range (2) loop
            Model.Wz_In (I, J) := Model.Wz_In (I, J) - (Gradient.Wz_In (I, J) * LR);
        end loop;
    end loop;

    for I in Model.Wc_In'Range (1) loop
        for J in Model.Wc_In'Range (2) loop
            Model.Wc_In (I, J) := Model.Wc_In (I, J) - (Gradient.Wc_In (I, J) * LR);
        end loop;
    end loop;

    -- Update Recurrent Weights

    for I in Model.Wr_Rec'Range (1) loop
        for J in Model.Wr_Rec'Range (2) loop
            Model.Wr_Rec (I, J) := Model.Wr_Rec (I, J) - (Gradient.Wr_Rec (I, J) * LR);
        end loop;
    end loop;

    for I in Model.Wz_Rec'Range (1) loop
        for J in Model.Wz_Rec'Range (2) loop
            Model.Wz_Rec (I, J) := Model.Wz_Rec (I, J) - (Gradient.Wz_Rec (I, J) * LR);
        end loop;
    end loop;

    for I in Model.Wc_Rec'Range (1) loop
        for J in Model.Wc_Rec'Range (2) loop
            Model.Wc_Rec (I, J) := Model.Wc_Rec (I, J) - (Gradient.Wc_Rec (I, J) * LR);
        end loop;
    end loop;

    -- Update Input Bias

    for I in Model.Br_In'Range loop
        Model.Br_In (I) := Model.Br_In (I) - (Gradient.Br_In (I) * LR);
    end loop;

    for I in Model.Bz_In'Range loop
        Model.Bz_In (I) := Model.Bz_In (I) - (Gradient.Bz_In (I) * LR);
    end loop;

    for I in Model.Bc_In'Range loop
        Model.Bc_In (I) := Model.Bc_In (I) - (Gradient.Bc_In (I) * LR);
    end loop;

    -- Update Output Weights

    for I in Model.W_Out'Range (1) loop
        for J in Model.W_Out'Range (2) loop
            Model.W_Out (I, J) := Model.W_Out (I, J) - (Gradient.W_Out (I, J) * LR);
        end loop;
    end loop;

    -- Update Output Bias

    for I in Model.B_Out'Range loop
        Model.B_Out (I) := Model.B_Out (I) - (Gradient.B_Out (I) * LR);
    end loop;

end Weight_Update_SGD;
