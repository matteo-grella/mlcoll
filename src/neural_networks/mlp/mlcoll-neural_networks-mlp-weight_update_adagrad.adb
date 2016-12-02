------------------------------------------------------------------------------
--                               M L C O L L
--  M a c h i n e   L e a r n i n g   C o m p o n e n t   C o l l e c t i o n
--
--     Copyright 2014 M. Grella, S. Cangialosi, E. Brambilla, M. Nicola
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

separate (MLColl.Neural_Networks.MLP)

procedure Weight_Update_ADAGRAD
  (Model            : in out Model_Type;
   Gradient         : in     Gradient_Type) is

    LR    : Real renames Model.Learning_Rate;

    Epsilon   : Real renames Model.Configuration.ADAM_Hypermarams.Epsilon;
begin

    -- Update Input Weight
    for I in Model.W_In'Range (1) loop
        if Gradient.Input_Non_Zero(I) then -- optimization
            for J in Model.W_In'Range (2) loop
                Model.Wv_In (I, J) := Model.Wv_In (I, J) + (Gradient.W_In (I, J) * Gradient.W_In (I, J));

                Model.W_In (I, J)  := Model.W_In (I, J) - (LR * Gradient.W_In(I, J) / (Sqrt(Model.Wv_In (I, J)) + Epsilon));
            end loop;
        end if;
    end loop;

    -- Update Input Bias
    for I in Model.B_In'Range loop
        declare
            Delta_GB_In : constant Real := Gradient.B_In (I);
        begin
            Model.Bv_In (I) :=  Model.Bv_In (I) + (Delta_GB_In * Delta_GB_In);
            Model.B_In (I) := Model.B_In (I)
              - (LR * Delta_GB_In / (Sqrt(Model.Bv_In (I)) + Epsilon) );
        end;
    end loop;

    -- Update Output Weight Layer
    for I in Model.W_Out'Range (1) loop
        for J in Model.W_Out'Range (2) loop
            declare
                Delta_GW_Out : constant Real := Gradient.W_Out (I, J);
            begin
                Model.Wv_Out (I, J) := Model.Wv_Out (I, J) + (Delta_GW_Out * Delta_GW_Out);
                Model.W_Out (I, J)  := Model.W_Out (I, J)
                  - (LR * Delta_GW_Out /  (Sqrt(Model.Wv_Out (I, J)) + Epsilon) );
            end;
        end loop;
    end loop;

    -- Update Output Bias
    for I in Model.B_Out'Range loop
        declare
            Delta_GB_Out : constant Real := Gradient.B_Out (I);
        begin
            Model.Bv_Out (I) := Model.Bv_Out (I) + (Delta_GB_Out * Delta_GB_Out);
            Model.B_Out (I)  := Model.B_Out (I)
              - (LR * Delta_GB_Out /  (Sqrt(Model.Bv_Out (I)) + Epsilon) );
        end;
    end loop;

end Weight_Update_ADAGRAD;
