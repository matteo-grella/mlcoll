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

separate (MLColl.Neural_Networks.Recurrent.GRU_With_Attention)

procedure Weight_Update_ADAM
  (Model      : in out Model_Type;
   Gradient   : in     Gradient_Type) is

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

    for I in Model.Wcm_In'Range (1) loop
        for J in Model.Wcm_In'Range (2) loop
            declare
                Delta_GW_In : constant Real := Gradient.Wc_In (I, J);
            begin
                Model.Wcm_In (I, J) := (Beta1 * Model.Wcm_In (I, J)) + (Beta1_Inv * Delta_GW_In);
                Model.Wcv_In (I, J) := (Beta2 * Model.Wcv_In (I, J)) + (Beta2_Inv * Delta_GW_In * Delta_GW_In);
                Model.Wc_In (I, J)  := Model.Wc_In (I, J)
                  - ( (Alpha * Model.Wcm_In (I, J)) /  (Sqrt((Model.Wcv_In (I, J))) + Epsilon) );
            end;
        end loop;
    end loop;

    for I in Model.Wzm_In'Range (1) loop
        for J in Model.Wzm_In'Range (2) loop
            declare
                Delta_GW_In : constant Real := Gradient.Wz_In (I, J);
            begin
                Model.Wzm_In (I, J) := (Beta1 * Model.Wzm_In (I, J)) + (Beta1_Inv * Delta_GW_In);
                Model.Wzv_In (I, J) := (Beta2 * Model.Wzv_In (I, J)) + (Beta2_Inv * Delta_GW_In * Delta_GW_In);
                Model.Wz_In (I, J)  := Model.Wz_In (I, J)
                  - ( (Alpha * Model.Wzm_In (I, J)) /  (Sqrt((Model.Wzv_In (I, J))) + Epsilon) );
            end;
        end loop;
    end loop;

    for I in Model.Wrm_In'Range (1) loop
        for J in Model.Wrm_In'Range (2) loop
            declare
                Delta_GW_In : constant Real := Gradient.Wr_In (I, J);
            begin
                Model.Wrm_In (I, J) := (Beta1 * Model.Wrm_In (I, J)) + (Beta1_Inv * Delta_GW_In);
                Model.Wrv_In (I, J) := (Beta2 * Model.Wrv_In (I, J)) + (Beta2_Inv * Delta_GW_In * Delta_GW_In);
                Model.Wr_In (I, J)  := Model.Wr_In (I, J)
                  - ( (Alpha * Model.Wrm_In (I, J)) /  (Sqrt((Model.Wrv_In (I, J))) + Epsilon) );
            end;
        end loop;
    end loop;

    -- Update Recurrent Hidden Weight

    for I in Model.Wcm_Rec'Range (1) loop
        for J in Model.Wcm_Rec'Range (2) loop
            declare
                Delta_GW_Rec : constant Real := Gradient.Wc_Rec (I, J);
            begin
                Model.Wcm_Rec (I, J) := (Beta1 * Model.Wcm_Rec (I, J)) + (Beta1_Inv * Delta_GW_Rec);
                Model.Wcv_Rec (I, J) := (Beta2 * Model.Wcv_Rec (I, J)) + (Beta2_Inv * Delta_GW_Rec * Delta_GW_Rec);
                Model.Wc_Rec (I, J)  := Model.Wc_Rec (I, J)
                  - ( (Alpha * Model.Wcm_Rec (I, J)) /  (Sqrt((Model.Wcv_Rec (I, J))) + Epsilon) );
            end;

        end loop;
    end loop;

    for I in Model.Wzm_Rec'Range (1) loop
        for J in Model.Wzm_Rec'Range (2) loop
            declare
                Delta_GW_Rec : constant Real := Gradient.Wz_Rec (I, J);
            begin
                Model.Wzm_Rec (I, J) := (Beta1 * Model.Wzm_Rec (I, J)) + (Beta1_Inv * Delta_GW_Rec);
                Model.Wzv_Rec (I, J) := (Beta2 * Model.Wzv_Rec (I, J)) + (Beta2_Inv * Delta_GW_Rec * Delta_GW_Rec);
                Model.Wz_Rec (I, J)  := Model.Wz_Rec (I, J)
                  - ( (Alpha * Model.Wzm_Rec (I, J)) /  (Sqrt((Model.Wzv_Rec (I, J))) + Epsilon) );
            end;

        end loop;
    end loop;

    for I in Model.Wrm_Rec'Range (1) loop
        for J in Model.Wrm_Rec'Range (2) loop
            declare
                Delta_GW_Rec : constant Real := Gradient.Wr_Rec (I, J);
            begin
                Model.Wrm_Rec (I, J) := (Beta1 * Model.Wrm_Rec (I, J)) + (Beta1_Inv * Delta_GW_Rec);
                Model.Wrv_Rec (I, J) := (Beta2 * Model.Wrv_Rec (I, J)) + (Beta2_Inv * Delta_GW_Rec * Delta_GW_Rec);
                Model.Wr_Rec (I, J)  := Model.Wr_Rec (I, J)
                  - ( (Alpha * Model.Wrm_Rec (I, J)) /  (Sqrt((Model.Wrv_Rec (I, J))) + Epsilon) );
            end;

        end loop;
    end loop;

    -- Update Input Bias

    for I in Model.Bc_In'Range loop
        declare
            Delta_GB_In : constant Real := Gradient.Bc_In (I);
        begin
            Model.Bcm_In (I) := (Beta1 * Model.Bcm_In (I)) + (Beta1_Inv * Delta_GB_In );
            Model.Bcv_In (I) := (Beta2 * Model.Bcv_In (I)) + (Beta2_Inv * Delta_GB_In * Delta_GB_In  );
            Model.Bc_In (I) := Model.Bc_In (I)
              - ( (Alpha * Model.Bcm_In (I)) /  (Sqrt(Model.Bcv_In (I)) + Epsilon) );
        end;
    end loop;

    for I in Model.Bz_In'Range loop
        declare
            Delta_GB_In : constant Real := Gradient.Bz_In (I);
        begin
            Model.Bzm_In (I) := (Beta1 * Model.Bzm_In (I)) + (Beta1_Inv * Delta_GB_In );
            Model.Bzv_In (I) := (Beta2 * Model.Bzv_In (I)) + (Beta2_Inv * Delta_GB_In * Delta_GB_In  );
            Model.Bz_In (I) := Model.Bz_In (I)
              - ( (Alpha * Model.Bzm_In (I)) /  (Sqrt(Model.Bzv_In (I)) + Epsilon) );
        end;
    end loop;

    for I in Model.Br_In'Range loop
        declare
            Delta_GB_In : constant Real := Gradient.Br_In (I);
        begin
            Model.Brm_In (I) := (Beta1 * Model.Brm_In (I)) + (Beta1_Inv * Delta_GB_In );
            Model.Brv_In (I) := (Beta2 * Model.Brv_In (I)) + (Beta2_Inv * Delta_GB_In * Delta_GB_In  );
            Model.Br_In (I) := Model.Br_In (I)
              - ( (Alpha * Model.Brm_In (I)) /  (Sqrt(Model.Brv_In (I)) + Epsilon) );
        end;
    end loop;

    -- Update Output Weight

    for I in Model.Wm_Out'Range (1) loop
        for J in Model.Wm_Out'Range (2) loop
            declare
                Delta_GW_Out : constant Real := Gradient.W_Out (I, J);
            begin
                Model.Wm_Out (I, J) := (Beta1 * Model.Wm_Out (I, J)) + (Beta1_Inv * Delta_GW_Out);
                Model.Wv_Out (I, J) := (Beta2 * Model.Wv_Out (I, J)) + (Beta2_Inv * Delta_GW_Out * Delta_GW_Out);
                Model.W_Out (I, J)  := Model.W_Out (I, J)
                  - ( (Alpha * Model.Wm_Out (I, J)) /  (Sqrt((Model.Wv_Out (I, J))) + Epsilon) );
            end;
        end loop;
    end loop;

    for I in Model.Bm_Out'Range loop
        declare
            Delta_GB_Out : constant Real := Gradient.B_Out (I);
        begin
            Model.Bm_Out (I) := (Beta1 * Model.Bm_Out (I)) + (Beta1_Inv * Delta_GB_Out);
            Model.Bv_Out (I) := (Beta2 * Model.Bv_Out (I)) + (Beta2_Inv * Delta_GB_Out * Delta_GB_Out);
            Model.B_Out (I)  := Model.B_Out (I)
              - ( (Alpha * Model.Bm_Out (I)) /  (Sqrt((Model.Bv_Out (I))) + Epsilon) );
        end;
    end loop;

    -- Update Attention Layer

    for I in Model.Ba_In'Range loop
        declare
            Delta_GBa_In : constant Real := Gradient.Ba_In (I);
        begin
            Model.Bam_In (I) := (Beta1 * Model.Bam_In (I)) + (Beta1_Inv * Delta_GBa_In);
            Model.Bav_In (I) := (Beta2 * Model.Bav_In (I)) + (Beta2_Inv * Delta_GBa_In * Delta_GBa_In);
            Model.Ba_In (I) := Model.Ba_In (I)
              - ( (Alpha * Model.Bam_In (I)) /  (Sqrt(Model.Bav_In (I)) + Epsilon) );
        end;
    end loop;

    for I in Model.A_In'Range loop
        declare
            Delta_GA_In : constant Real := Gradient.A_In (I);
        begin
            Model.Am_In (I) := (Beta1 * Model.Am_In (I)) + (Beta1_Inv * Delta_GA_In);
            Model.Av_In (I) := (Beta2 * Model.Av_In (I)) + (Beta2_Inv * Delta_GA_In * Delta_GA_In);
            Model.A_In (I) := Model.A_In (I)
              - ( (Alpha * Model.Am_In (I)) /  (Sqrt(Model.Av_In (I)) + Epsilon) );
        end;
    end loop;

    for I in Model.Wam_In'Range (1) loop
        for J in Model.Wam_In'Range (2) loop
            declare
                Delta_GWa_In : constant Real := Gradient.Wa_In (I, J);
            begin
                Model.Wam_In (I, J) := (Beta1 * Model.Wam_In (I, J)) + (Beta1_Inv * Delta_GWa_In);
                Model.Wav_In (I, J) := (Beta2 * Model.Wav_In (I, J)) + (Beta2_Inv * Delta_GWa_In * Delta_GWa_In);
                Model.Wa_In (I, J)  := Model.Wa_In (I, J)
                  - ( (Alpha * Model.Wam_In (I, J)) /  (Sqrt((Model.Wav_In (I, J))) + Epsilon) );
            end;
        end loop;
    end loop;

    for I in Model.Wbm_In'Range (1) loop
        for J in Model.Wbm_In'Range (2) loop
            declare
                Delta_GWb_In : constant Real := Gradient.Wb_In (I, J);
            begin
                Model.Wbm_In (I, J) := (Beta1 * Model.Wbm_In (I, J)) + (Beta1_Inv * Delta_GWb_In);
                Model.Wbv_In (I, J) := (Beta2 * Model.Wbv_In (I, J)) + (Beta2_Inv * Delta_GWb_In * Delta_GWb_In);
                Model.Wb_In (I, J)  := Model.Wb_In (I, J)
                  - ( (Alpha * Model.Wbm_In (I, J)) /  (Sqrt((Model.Wbv_In (I, J))) + Epsilon) );
            end;
        end loop;
    end loop;

    for I in Model.Wam_Rec'Range (1) loop
        for J in Model.Wam_Rec'Range (2) loop
            declare
                Delta_GWa_Rec : constant Real := Gradient.Wa_Rec (I, J);
            begin
                Model.Wam_Rec (I, J) := (Beta1 * Model.Wam_Rec (I, J)) + (Beta1_Inv * Delta_GWa_Rec);
                Model.Wav_Rec (I, J) := (Beta2 * Model.Wav_Rec (I, J)) + (Beta2_Inv * Delta_GWa_Rec * Delta_GWa_Rec);
                Model.Wa_Rec (I, J)  := Model.Wa_Rec (I, J)
                  - ( (Alpha * Model.Wam_Rec (I, J)) /  (Sqrt((Model.Wav_Rec (I, J))) + Epsilon) );
            end;
        end loop;
    end loop;


end Weight_Update_ADAM;
