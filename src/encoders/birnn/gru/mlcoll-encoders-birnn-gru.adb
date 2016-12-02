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

with Ada.Text_IO; use Ada.Text_IO;

with MLColl.Neural_Networks; use MLColl.Neural_Networks;

package body MLColl.Encoders.BiRNN.GRU is

    overriding procedure Initialize_BiRNN_Model
      (Model             : in out GRU_Model_Type;
       Input_Layer_Size  : in Positive;
       Hidden_Layer_Size : in Positive;
       Output_Layer_Size : in Positive;
       Verbose           : in Boolean := False) is

        GRU_Configuration : constant GRU.Configuration_Type
          := (Input_Layer_Size                => Input_Layer_Size,
              Hidden_Layer_Size               => Hidden_Layer_Size,
              Output_Layer_Size               => Output_Layer_Size / 2,
              Initial_Learning_Rate           => 0.0001,
              Random_Weights_Range            => 0.1,
              Max_Num_Of_Epochs               => 0,
              Min_Num_Of_Epochs               => 0,
              Max_No_Best_Epochs              => 0,
              Activation_Function_Name        => Tanh,
              Leaky_ReLU_Factor               => 0.01,
              Learning_Rule                   => ADAM,
              ADAM_Hypermarams                => Default_ADAM_Hyperparams,
              Propagate_Error_To_Input_Layer  => True);
    begin

        Model.Input_Layer_Size  := Input_Layer_Size;
        Model.Hidden_Layer_Size := Hidden_Layer_Size;
        Model.Output_Layer_Size := Output_Layer_Size;

        for RNN_Direction in RNN_Direction_Type'Range loop
            GRU.Initialize
              (Model              => Model.RNN_Models (RNN_Direction),
               Configuration      => GRU_Configuration,
               Initialize_Weights => True);
        end loop;

        if Verbose then
            New_Line;
            Put_Line ("[L2R_RNN]     Learning rate: " & Model.RNN_Models (L2R).Learning_Rate'Img);
            Put_Line ("[L2R_RNN]  Input_Layer_Size: " & GRU.Get_Input_Layer_Size (Model.RNN_Models (L2R))'Img);
            Put_Line ("[L2R_RNN] Hidden_Layer_Size: " & GRU.Get_Hidden_Layer_Size (Model.RNN_Models (L2R))'Img);
            Put_Line ("[L2R_RNN] Output_Layer_Size: " & GRU.Get_Output_Layer_Size (Model.RNN_Models (L2R))'Img);
            New_Line;
            Put_Line ("[L2R_RNN]     Learning rate: " & Model.RNN_Models (R2L).Learning_Rate'Img);
            Put_Line ("[R2L_RNN]  Input_Layer_Size: " & GRU.Get_Input_Layer_Size (Model.RNN_Models (L2R))'Img);
            Put_Line ("[R2L_RNN] Hidden_Layer_Size: " & GRU.Get_Hidden_Layer_Size (Model.RNN_Models (L2R))'Img);
            Put_Line ("[R2L_RNN] Output_Layer_Size: " & GRU.Get_Output_Layer_Size (Model.RNN_Models (L2R))'Img);
            New_Line;
        end if;

        Model.Initialized := True;

    end Initialize_BiRNN_Model;

    overriding procedure Finalize_BiRNN_Model
      (Model : in out GRU_Model_Type) is
    begin
        for Item of Model.RNN_Models loop
            GRU.Finalize (Item);
        end loop;

        Model.Initialized := False;
    end Finalize_BiRNN_Model;

    overriding procedure Initialize_BiRNN_Structure
      (BiRNN_Structure   : in out BiRNN_GRU_Structure_Type;
       Input_Layer_Size  : in Positive;
       Hidden_Layer_Size : in Positive;
       Output_Layer_Size : in Positive;
       Input_Sequence    : in Encoded_Entry_Array_Type) is
    begin

        for T in Input_Sequence'Range loop
            BiRNN_Structure.Input_Sequence (Index_Type (T))
              := new Real_Array (Input_Sequence (T)'Range);

            for J in Input_Sequence (T)'Range loop
                BiRNN_Structure.Input_Sequence (Index_Type (T)) (J) := Input_Sequence (T) (J).all;
            end loop;

            BiRNN_Structure.Input_Sequence_Gradients (Index_Type (T))
              := new Real_Array (Input_Sequence (T)'Range);

            BiRNN_Structure.Encoded_Sequence (Index_Type (T)) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Output_Layer_Size) - 1);

            BiRNN_Structure.Sequence_Output_Error (Index_Type (T))
              := new Real_Array (BiRNN_Structure.Encoded_Sequence (Index_Type (T))'Range);
        end loop;

        for RNN_Direction in RNN_Direction_Type'Range loop

            BiRNN_Structure.RNN_Structure (RNN_Direction)
              := new GRU.GRU_Structure_Type
                (First_Sequence_Index        => BiRNN_Structure.First_Sequence_Index,
                 Last_Sequence_Index         => BiRNN_Structure.Last_Sequence_Index,
                 First_Hidden_Sequence_Index => BiRNN_Structure.First_Hidden_Sequence_Index,
                 Input_Layer_Size            => Input_Layer_Size,
                 Hidden_Layer_Size           => Hidden_Layer_Size,
                 Output_Layer_Size           => Output_Layer_Size / 2,
                 Input_Layer_Last            => Index_Type'First + Index_Type (Input_Layer_Size) - 1,
                 Hidden_Layer_Last           => Index_Type'First + Index_Type (Hidden_Layer_Size) - 1,
                 Output_Layer_Last           => Index_Type'First + Index_Type (Output_Layer_Size / 2) - 1);

            GRU.Initialize_GRU_Structure
              (GRU_Structure               => BiRNN_Structure.RNN_Structure (RNN_Direction).all);
        end loop;

        Fill
          (V_In       => BiRNN_Structure.Input_Sequence,
           V1_Out     => BiRNN_Structure.RNN_Structure (L2R).Sequence_Input,
           V2_Rev_Out => BiRNN_Structure.RNN_Structure (R2L).Sequence_Input);

        BiRNN_Structure.Initialized := True;

    end Initialize_BiRNN_Structure;

    overriding procedure Finalize_BiRNN_Structure
      (BiRNN_Structure  : in out BiRNN_GRU_Structure_Type) is
    begin

        Free (BiRNN_Structure.Input_Sequence);
        Free (BiRNN_Structure.Input_Sequence_Gradients);
        Free (BiRNN_Structure.Encoded_Sequence);
        Free (BiRNN_Structure.Sequence_Output_Error);

        for RNN_Direction in RNN_Direction_Type'Range loop
            GRU.Finalize_GRU_Structure
              (GRU_Structure => BiRNN_Structure.RNN_Structure (RNN_Direction).all);

            Free (BiRNN_Structure.RNN_Structure (RNN_Direction));
        end loop;

        BiRNN_Structure.Initialized := False;
    end Finalize_BiRNN_Structure;

    overriding procedure Encode
      (Model           : in     GRU_Model_Type;
       BiRNN_Structure : in out BiRNN_Structure_Type'Class;
       Verbose         : in     Boolean := False) is
    begin

        for RNN_Direction in RNN_Direction_Type'Range loop
            if Verbose then
                Put_Line (Standard_Error, "[" & RNN_Direction'Img & "] Forward");
            end if;

            GRU.Forward
              (Model         => Model.RNN_Models (RNN_Direction),
               GRU_Structure => BiRNN_GRU_Structure_Type (BiRNN_Structure).RNN_Structure (RNN_Direction));
        end loop;

        if Verbose then
            Put_Line (Standard_Error, "Concatenate Bidirectional RNNs output");
        end if;

        Concatenate
          (V_Out  => BiRNN_Structure.Encoded_Sequence,
           V1_In  => BiRNN_GRU_Structure_Type (BiRNN_Structure).RNN_Structure (L2R).Sequence_Output,
           V2_In  => BiRNN_GRU_Structure_Type (BiRNN_Structure).RNN_Structure (R2L).Sequence_Output);

    end Encode;

    overriding procedure Learn
      (Model           : in out GRU_Model_Type;
       BiRNN_Structure : in out BiRNN_Structure_Type'Class;
       Verbose         : in     Boolean := False) is
    begin

        if Verbose then
            Put_Line (Standard_Error, "Divide BiRNN Output Error");
        end if;

        Partition
          (V_In   => BiRNN_Structure.Sequence_Output_Error,
           V1_Out => BiRNN_GRU_Structure_Type (BiRNN_Structure).RNN_Structure (L2R).Sequence_Output_Error,
           V2_Out => BiRNN_GRU_Structure_Type (BiRNN_Structure).RNN_Structure (R2L).Sequence_Output_Error);


        for RNN_Direction in RNN_Direction_Type'Range loop

            if Verbose then
                Put_Line (Standard_Error, "[" & RNN_Direction'Img & "] Backward");
            end if;

            GRU.Backward
              (Model          => Model.RNN_Models (RNN_Direction),
               GRU_Structure  => BiRNN_GRU_Structure_Type (BiRNN_Structure).RNN_Structure(RNN_Direction).all);

            if Verbose then
                Put_Line (Standard_Error, "[" & RNN_Direction'Img & "]  Weight Update");
            end if;

            GRU.Weight_Update
              (Model    => Model.RNN_Models (RNN_Direction),
               Gradient => BiRNN_GRU_Structure_Type (BiRNN_Structure).RNN_Structure (RNN_Direction).Gradient);

        end loop;

        if Verbose then
            Put_Line (Standard_Error, "Accumulate RNN Errors to BiRNN Sequence Input Gradients");
        end if;

        Merge
          (V_Out => BiRNN_Structure.Input_Sequence_Gradients ,
           V1_In => BiRNN_GRU_Structure_Type (BiRNN_Structure).RNN_Structure(L2R).Sequence_Input_Gradients,
           V2_In => BiRNN_GRU_Structure_Type (BiRNN_Structure).RNN_Structure(R2L).Sequence_Input_Gradients);

    end Learn;

end MLColl.Encoders.BiRNN.GRU;
