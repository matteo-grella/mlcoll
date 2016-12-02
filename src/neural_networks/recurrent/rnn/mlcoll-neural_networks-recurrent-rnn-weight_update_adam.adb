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

procedure Weight_Update_ADAM
  (Model            : in out Model_Type;
   Gradient         : in     Gradient_Type) is

    LR    : Real renames Model.Learning_Rate;
    Alpha : Real;

    Beta1     : Real renames Model.Configuration.ADAM_Hypermarams.Beta1;
    Beta2     : Real renames Model.Configuration.ADAM_Hypermarams.Beta2;
    Beta1_Inv : Real renames Model.Configuration.ADAM_Hypermarams.Beta1_Inv;
    Beta2_Inv : Real renames Model.Configuration.ADAM_Hypermarams.Beta2_Inv;
    Epsilon   : Real renames Model.Configuration.ADAM_Hypermarams.Epsilon;
begin

    Model.Timestep := Model.Timestep + 1.0;

    Alpha := LR * Real(Sqrt(1.0 - (Beta2 ** Natural(Model.Timestep)))) / (1.0 - (Beta1 ** Natural(Model.Timestep)));

    -- Update Input Weight
    for I in Model.W_In'Range (1) loop
        if Gradient.Input_Non_Zero(I) then -- optimization
            for J in Model.W_In'Range (2) loop
                Model.Wm_In (I, J) := (Beta1 * Model.Wm_In (I, J)) + (Beta1_Inv * Gradient.W_In(I, J));
                Model.Wv_In (I, J) := (Beta2 * Model.Wv_In (I, J)) + (Beta2_Inv * Gradient.W_In(I, J) * Gradient.W_In(I, J));
                Model.W_In (I, J)  := Model.W_In (I, J) - ( (Alpha * Model.Wm_In (I, J)) /  (Sqrt(Model.Wv_In (I, J)) + Epsilon) );
            end loop;
        end if;
    end loop;

    -- Update Recurrent Weight
    for I in Model.Wm_Rec'Range (1) loop
        for J in Model.Wm_Rec'Range (2) loop
            declare
                Delta_GW_Rec : constant Real := Gradient.W_Rec (I, J);
            begin
                Model.Wm_Rec (I, J) := (Beta1 * Model.Wm_Rec (I, J)) + (Beta1_Inv * Delta_GW_Rec);
                Model.Wv_Rec (I, J) := (Beta2 * Model.Wv_Rec (I, J)) + (Beta2_Inv * Delta_GW_Rec * Delta_GW_Rec);
                Model.W_Rec (I, J)  := Model.W_Rec (I, J)
                  - ( (Alpha * Model.Wm_Rec (I, J)) /  (Sqrt ((Model.Wv_Rec (I, J))) + Epsilon) );
            end;
        end loop;
    end loop;

    -- Update Input Bias
    for I in Model.Bm_In'Range loop
        declare
            Delta_GB_In : constant Real := Gradient.B_In (I);
        begin
            Model.Bm_In (I) := (Beta1 * Model.Bm_In (I)) + (Beta1_Inv * Delta_GB_In );
            Model.Bv_In (I) := (Beta2 * Model.Bv_In (I)) + (Beta2_Inv * Delta_GB_In * Delta_GB_In  );
            Model.B_In (I) := Model.B_In (I)
              - ( (Alpha * Model.Bm_In (I)) /  (Sqrt(Model.Bv_In (I)) + Epsilon) );
        end;
    end loop;

    -- Update Output Weight Layer
    for I in Model.Wm_Out'Range (1) loop
        for J in Model.Wm_Out'Range (2) loop
            declare
                Delta_GW_Out : constant Real := Gradient.W_Out (I, J);
            begin
                Model.Wm_Out (I, J) := (Beta1 * Model.Wm_Out (I, J)) + (Beta1_Inv * Delta_GW_Out);
                Model.Wv_Out (I, J) := (Beta2 * Model.Wv_Out (I, J)) + (Beta2_Inv * Delta_GW_Out * Delta_GW_Out);
                Model.W_Out (I, J)  := Model.W_Out (I, J)
                  - ( (Alpha * Model.Wm_Out (I, J)) /  (Sqrt(Model.Wv_Out (I, J)) + Epsilon) );
            end;
        end loop;
    end loop;

    -- Update Output Bias
    for I in Model.Bm_Out'Range loop
        declare
            Delta_GB_Out : constant Real := Gradient.B_Out (I);
        begin
            Model.Bm_Out (I) := (Beta1 * Model.Bm_Out (I)) + (Beta1_Inv * Delta_GB_Out);
            Model.Bv_Out (I) := (Beta2 * Model.Bv_Out (I)) + (Beta2_Inv * Delta_GB_Out * Delta_GB_Out);
            Model.B_Out (I)  := Model.B_Out (I)
              - ( (Alpha * Model.Bm_Out (I)) /  (Sqrt(Model.Bv_Out (I)) + Epsilon) );
        end;
    end loop;

end Weight_Update_ADAM;
