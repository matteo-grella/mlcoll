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

procedure Add_L2_Regularization
  (Model       : in out Model_Type;
   Gradient    : in out Gradient_Type) is

    Regularization_Weight : Real renames Model.Configuration.Regularization_Parameter;
begin

    -- Input Weight
    for I in Model.W_In'Range (1) loop
        for J in Model.W_In'Range (2) loop
            Gradient.W_In (I, J) := Gradient.W_In (I, J) +
              (Regularization_Weight * Model.W_In (I, J));
        end loop;
    end loop;
        
    -- Input Bias
    for I in Model.B_In'Range loop
        Gradient.B_In (I) := Gradient.B_In (I) +
          (Regularization_Weight * Model.B_In (I));
    end loop;
               
    -- Output Weight
    for I in Model.W_Out'Range (1) loop
        for J in Model.W_Out'Range (2) loop
            Gradient.W_Out (I, J) := Gradient.W_Out (I, J) +
              (Regularization_Weight * Model.W_Out (I, J));
        end loop;
    end loop;
        
    -- Output Bias
    for I in Model.B_Out'Range loop
        Gradient.B_Out (I) := Gradient.B_Out (I) +
          (Regularization_Weight * Model.B_Out (I));
    end loop;
        
end Add_L2_Regularization;
