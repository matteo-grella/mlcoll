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

private with Ada.Streams.Stream_IO;
with Ada.Text_IO; use Ada.Text_IO;

with ARColl.Numerics.Reals.Functions; use ARColl.Numerics.Reals.Functions;

package body MLColl.Neural_Networks.CNN is

    package Stream_IO renames Ada.Streams.Stream_IO;

    function Make_CNN_Structure
      (First_Sequence_Index : in Index_Type;
       Last_Sequence_Index  : in Index_Type;
       Input_Layer_Size     : in Positive;
       Hidden_Layer_Size    : in Positive;
       Output_Layer_Size    : in Positive) return CNN_Structure_Type is

        CNN_Structure : CNN_Structure_Type
          (First_Sequence_Index    => First_Sequence_Index,
           Last_Sequence_Index     => Last_Sequence_Index,
           Input_Layer_Size        => Input_Layer_Size,
           Hidden_Layer_Size       => Hidden_Layer_Size,
           Output_Layer_Size       => Output_Layer_Size,
           Input_Layer_Last        => Index_Type'First + Index_Type (Input_Layer_Size) - 1,
           Hidden_Layer_Last       => Index_Type'First + Index_Type(Hidden_Layer_Size) - 1,
           Output_Layer_Last       => Index_Type'First + Index_Type(Output_Layer_Size) - 1);
    begin
        return CNN_Structure;
    end Make_CNN_Structure;

    function Make_Gradient
      (Input_Layer_Size    : in     Positive;
       Hidden_Layer_Size   : in     Positive;
       Output_Layer_Size   : in     Positive) return Gradient_Type is

        Gradient : Gradient_Type
          (Input_Layer_Last        => Index_Type'First + Index_Type (Input_Layer_Size) - 1,
           Hidden_Layer_Last       => Index_Type'First + Index_Type(Hidden_Layer_Size) - 1,
           Output_Layer_Last       => Index_Type'First + Index_Type(Output_Layer_Size) - 1);
    begin
        return Gradient;
    end Make_Gradient;

    procedure Initialize_CNN_Structure
      (CNN_Structure      : in out CNN_Structure_Type) is

        Input_Layer_Size   : Positive renames CNN_Structure.Input_Layer_Size;
        Hidden_Layer_Size  : Positive renames CNN_Structure.Hidden_Layer_Size;
        Output_Layer_Size  : Positive renames CNN_Structure.Output_Layer_Size;

    begin
        if Hidden_Layer_Size /= Output_Layer_Size then
            raise CNN_Exception
              with "CNN Error: Hidden_Layer_Size'Length /= Output_Layer_Size'Length";
        end if;

        CNN_Structure.Output_Error          := new Real_Array (Index_Type'First .. Index_Type'First + Index_Type (Output_Layer_Size) -1);
        CNN_Structure.Output_Layer          := new Real_Array (Index_Type'First .. Index_Type'First + Index_Type (Output_Layer_Size) -1);
        CNN_Structure.Output_Layer_Argmax   := new Index_Type_Array (Index_Type'First .. Index_Type'First + Index_Type (Output_Layer_Size) -1);
        CNN_Structure.Output_Error.all      := (others => Real'First); -- Max Pooling
        CNN_Structure.Output_Layer.all      := (others => 0.0);

        for T in CNN_Structure.First_Sequence_Index .. CNN_Structure.Last_Sequence_Index loop
            CNN_Structure.Sequence_Input(T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Input_Layer_Size) -  1);

            CNN_Structure.Sequence_Input_Gradients(T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Input_Layer_Size) -  1);

            CNN_Structure.Sequence_Hidden(T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);

            CNN_Structure.Sequence_Hidden_Derivative(T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);
        end loop;

    end Initialize_CNN_Structure;

    procedure Finalize_CNN_Structure
      (CNN_Structure : in out CNN_Structure_Type) is
    begin
        Free (CNN_Structure.Sequence_Input);
        Free (CNN_Structure.Sequence_Input_Gradients);
        Free (CNN_Structure.Sequence_Hidden);
        Free (CNN_Structure.Sequence_Hidden_Derivative);
        Free (CNN_Structure.Output_Layer);
        Free (CNN_Structure.Output_Error);
        Free (CNN_Structure.Output_Layer_Argmax);
    end Finalize_CNN_Structure;

    procedure Forward
      (Model                   : in     Model_Type;
       CNN_Structure           : access CNN_Structure_Type) is separate;

    function Last_Outcome_Index
      (Model               : in Model_Type) return Index_Type is
      (Index_Type'First + Index_Type(Model.Configuration.Output_Layer_Size) - 1);

    function Get_Output_Layer_Size
      (Model  : in Model_Type) return Positive is
      (Model.Configuration.Output_Layer_Size);

    function Get_Input_Layer_Size
      (Model  : in Model_Type) return Positive is
      (Model.Configuration.Input_Layer_Size);

    function Get_Hidden_Layer_Size
      (Model  : in Model_Type) return Positive is
      (Model.Configuration.Hidden_Layer_Size);

    function Get_Max_No_Best_Epochs
      (Model  : in Model_Type) return Positive is (Model.Configuration.Max_No_Best_Epochs);

    function Get_Min_Num_Of_Epochs
      (Model  : in Model_Type) return Positive is (Model.Configuration.Min_Num_Of_Epochs);

    procedure Print
      (Model : in out Model_Type;
       File  : Text_IO.File_Type := Text_IO.Standard_Output) is
    begin
        Text_IO.Put_Line (File, "Hidden_Layer_Size:"        & Model.Configuration.Hidden_Layer_Size'Img);
        Text_IO.Put_Line (File, "Initial_Learning_Rate:"    & Model.Configuration.Initial_Learning_Rate'Img);
        Text_IO.Put_Line (File, "Random_Weights_Range:"     & Model.Configuration.Random_Weights_Range'Img);
        Text_IO.Put_Line (File, "Max_Num_Of_Epochs:"        & Model.Configuration.Max_Num_Of_Epochs'Img);
        Text_IO.Put_Line (File, "Min_Iterations:"           & Model.Configuration.Min_Num_Of_Epochs'Img);
        Text_IO.Put_Line (File, "Max_No_Best_Epochs:"       & Model.Configuration.Max_No_Best_Epochs'Img);
        Text_IO.Put_Line (File, "Learning_Rate_Decrease_Constant:" & Model.Configuration.Learning_Rate_Decrease_Constant'Img);
    end Print;

    procedure Initialize
      (Model                : in out Model_Type;
       Configuration        : in     Configuration_Type;
       Initialize_Weights   : in     Boolean := False) is
    begin

        if Model.Is_Initialized then
            raise CNN_Exception
              with "Model already initialized";
        end if;

        Model.Configuration  := Configuration;
        Model.Learning_Rate  := Model.Configuration.Initial_Learning_Rate;

        Model.Is_Initialized := True;

        if Initialize_Weights then
            Initialize_Matrices (Model);
        end if;

    end Initialize;

    procedure Finalize
      (Model : in out Model_Type) is
    begin

        if not Model.Is_Initialized then
            raise CNN_Exception
              with "Model not initialized";
        end if;

        Model.Learning_Rate := Model.Configuration.Initial_Learning_Rate;

        Free (Model.W_In);
        Free (Model.B_In);

        --ADAM

        if Model.Configuration.Learning_Rule = ADAM then
            Model.Timestep := 0.0;

            Free (Model.Wm_In);
            Free (Model.Bm_In);
            Free (Model.Wv_In);
            Free (Model.Bv_In);
        end if;

        Model.Is_Initialized := False;
    end Finalize;

    procedure Initialize_Matrices
      (Model : in out Model_Type) is separate;

    function Calculate_Output_Error
      (CNN_Structure                   : in out CNN_Structure_Type;
       Gold_Output_Layer               : in     Real_Array) return Real is

        Output_Error        : Real_Array renames CNN_Structure.Output_Error.all;
        Output              : Real_Array renames CNN_Structure.Output_Layer.all;

        Loss : Real := 0.0;
    begin
        -- TODO
        if Output_Error'Length /= Output'Length then
            raise CNN_Exception
              with "Calc_Output_Error: Output_Error'Length /= Actual_Output_Layer'Length";
        end if;

        Output_Error := Output;

        for J in Output'First .. Output'Last loop
            Output_Error (J) := Output (J) - Gold_Output_Layer (J);
            Loss := Loss + (0.5 * ((Gold_Output_Layer (J) - Output (J)) ** 2));
        end loop;

        Loss := Loss / Real (Output_Error'Length);

        return Loss;

    end Calculate_Output_Error;

    procedure Backward
      (Model                  : in out Model_Type;
       CNN_Structure          : access CNN_Structure_Type;
       Gradient               : in out Gradient_Type) is separate;

    procedure Weight_Update_SGD
      (Model            : in out Model_Type;
       Gradient         : in     Gradient_Type) is separate;

    procedure Weight_Update_ADAM
      (Model            : in out Model_Type;
       Gradient         : in     Gradient_Type) is separate;

    procedure Weight_Update
      (Model            : in out Model_Type;
       Gradient         : in     Gradient_Type) is
    begin
        case Model.Configuration.Learning_Rule is
            when SGD =>
                Weight_Update_SGD
                  (Model                => Model,
                   Gradient             => Gradient);

            when ADAM =>
                Weight_Update_ADAM
                  (Model                => Model,
                   Gradient             => Gradient);

            when ADAGRAD =>
                  raise Constraint_Error with "ADAGRAD not implemented";
        end case;
    end Weight_Update;

    procedure Reset_Gradients
      (Gradient : in out Gradient_Type) is
    begin
        Gradient.B_In  := (others => 0.0);
        Gradient.W_In  := (others => (others => 0.0));
        Gradient.Input_Non_Zero := (others => False);
    end Reset_Gradients;

    procedure Decrease_Learning_Rate
      (Model : in out Model_Type;
       Epoch : in     Positive) is
    begin
        case Model.Configuration.Learning_Rule is
            when SGD =>
                Model.Learning_Rate
                  := Model.Configuration.Initial_Learning_Rate /
                    (1.0 +
                       (Model.Configuration.Learning_Rate_Decrease_Constant *
                              Real (Epoch)));

                -- if Model.Learning_Rate /= Model.Learning_Rate_Final and then Epoch > 1 then
                --    --Model.Learning_Rate := Exp((Model.Configuration.Max_Num_Of_Epochs - Epoch));
                --    --exp(((iterations - iteration) * log(trainer.learning_rate) + log(trainer.learning_rate_final)) / (iterations - iteration + 1));
                -- end if;

            when others => null;
        end case;

    end Decrease_Learning_Rate;

    procedure Serialize
      (Model          : in Model_Type;
       Model_Filename : in String) is

        SFile : Stream_IO.File_Type;
        SAcc  : Stream_IO.Stream_Access;
    begin
        Stream_IO.Create (SFile, Stream_IO.Out_File, Model_Filename);
        SAcc := Stream_IO.Stream (SFile);

        Model_Type'Output (SAcc, Model);

        Stream_IO.Close (SFile);
    end Serialize;

    procedure Load
      (Model          : out Model_Type;
       Model_Filename : in  String) is

        SFile : Stream_IO.File_Type;
        SAcc  : Stream_IO.Stream_Access;
    begin
        Stream_IO.Open (SFile, Stream_IO.In_File, Model_Filename);
        SAcc := Stream_IO.Stream (SFile);

        Model := Model_Type'Input (SAcc);

        Stream_IO.Close (SFile);

        if not Model.Is_Initialized then
            raise CNN_Exception
              with "Loaded model is not initialized";
        end if;

    end Load;

end MLColl.Neural_Networks.CNN;
