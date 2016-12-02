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

with Text_IO;

package MLColl.Neural_Networks.CNN is

    type Configuration_Type is record

        Input_Layer_Size  : Positive;
        -- Number of Neuron in the Input Layer

        Hidden_Layer_Size : Positive := 100;
        -- Number of Neuron in the Hidden Layer
        -- (alternative names : Feature Detection Layer, Convolutional Layer)

        Output_Layer_Size : Positive := 100;
        -- Hidden_layer_size must be equal to Output_layer_size
        -- Number of Neuron in the Output Layer
        --(alternative names: Feature Pooling Layer)

        Features_Type     : Features_Type_Type := DENSE;
        -- Define the type of Input_Layer (binary/real values)

        Initial_Learning_Rate : Real;
        -- Initial learning rate

        Random_Weights_Range  : Real;
        -- See "Get_Random_Weight" function defined
        -- inside "Initialize_Matrices" procedure

        Max_Num_Of_Epochs     : Natural := 0;
        -- Maximum number of training Epochs

        Min_Num_Of_Epochs     : Natural := 0;
        -- Minimum number of training Epochs

        Max_No_Best_Epochs    : Natural := 0;
        -- Maximum number of Epochs without a new "best" validation
        -- before train termination

        Learning_Rate_Decrease_Constant : Real;
        -- Used for learning-rate decreasing:
        -- (initial_learning_rate) / (1 + (LRDC * Epoch))

        Activation_Function_Name : Activation_Function_Name_Type;

        Leaky_ReLU_Factor : Real := 0.01;

        Learning_Rule     : Learning_Rule_Type := SGD;

        Regularization_Parameter : Real := 1.0E-8;

        ADAM_Hypermarams         : ADAM_Hyperparams_Type
          := Default_ADAM_Hyperparams;

        Constraint_Hyperparams   : Constraint_Hyperparams_Type
          := Default_Constraint_Hyperparams;

    end record;

    type Model_Type is tagged;

    ----
    -- Structures
    ----

    type Gradient_Type
      (Input_Layer_Last  : Index_Type;
       Hidden_Layer_Last : Index_Type;
       Output_Layer_Last : Index_Type) is record

        B_In : Real_Array
          (Index_Type'First .. Hidden_Layer_Last)
          := (others => 0.0);

        -- Kernel Matrix
        W_In : Real_Matrix
          (Index_Type'First .. Input_Layer_Last,
           Index_Type'First .. Hidden_Layer_Last)
          := (others => (others => 0.0));

        Input_Non_Zero : Boolean_Array
          (Index_Type'First .. Input_Layer_Last)
          := (others => False);

        Count : Positive := 1;
    end record;

    type CNN_Structure_Type
      (First_Sequence_Index        : Index_Type;
       Last_Sequence_Index         : Index_Type;
       Input_Layer_Size            : Positive;
       Hidden_Layer_Size           : Positive;
       Output_Layer_Size           : Positive;
       Input_Layer_Last            : Index_Type;
       Hidden_Layer_Last           : Index_Type;
       Output_Layer_Last           : Index_Type) is record

        Sequence_Input                  : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Input_Gradients        : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Hidden                 : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Hidden_Derivative      : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Output_Layer                    : Real_Array_Access;
        Output_Error                    : Real_Array_Access;
        Output_Layer_Argmax             : Index_Array_Access;
    end record;

    function Make_CNN_Structure
      (First_Sequence_Index : in Index_Type;
       Last_Sequence_Index  : in Index_Type;
       Input_Layer_Size     : in Positive;
       Hidden_Layer_Size    : in Positive;
       Output_Layer_Size    : in Positive) return CNN_Structure_Type;

    function Make_Gradient
      (Input_Layer_Size    : in     Positive;
       Hidden_Layer_Size   : in     Positive;
       Output_Layer_Size   : in     Positive) return Gradient_Type;

    ----
    -- Functions
    ----

    procedure Print
      (Model               : in out Model_Type;
       File                : Text_IO.File_Type := Text_IO.Standard_Output);
    -- Print configuration data values

    procedure Initialize
      (Model              : in out Model_Type;
       Configuration      : in     Configuration_Type;
       Initialize_Weights : in     Boolean := False);
    -- Initialize Model

    procedure Finalize
      (Model : in out Model_Type);
    -- Finalize Model

    procedure Load
      (Model          : out Model_Type;
       Model_Filename : in  String);
    -- Load Model from serialized file

    procedure Serialize
      (Model          : in Model_Type;
       Model_Filename : in String);
    -- Serialize Model to file

    ----------------
    -- Get Functions
    ----------------

    function First_Outcome_Index
      (Model               : in Model_Type) return Index_Type is
      (Index_Type'First) with Inline;

    function Last_Outcome_Index
      (Model               : in Model_Type) return Index_Type with Inline;

    function Get_Output_Layer_Size
      (Model  : in Model_Type) return Positive with Inline;

    function Get_Input_Layer_Size
      (Model  : in Model_Type) return Positive with Inline;

    function Get_Hidden_Layer_Size
      (Model  : in Model_Type) return Positive with Inline;

    function Get_Max_No_Best_Epochs
      (Model  : in Model_Type) return Positive with Inline;

    function Get_Min_Num_Of_Epochs
      (Model  : in Model_Type) return Positive with Inline;

    ----
    -- Network Functions
    ----

    procedure Initialize_Matrices
      (Model : in out Model_Type);
    -- Matrices initialization

    procedure Forward
      (Model                   : in     Model_Type;
       CNN_Structure           : access CNN_Structure_Type);

    procedure Reset_Gradients
      (Gradient : in out Gradient_Type);

    procedure Weight_Update_SGD
      (Model            : in out Model_Type;
       Gradient         : in     Gradient_Type);

    procedure Weight_Update_ADAM
      (Model            : in out Model_Type;
       Gradient         : in     Gradient_Type);

    procedure Weight_Update
      (Model            : in out Model_Type;
       Gradient         : in     Gradient_Type);

    function Calculate_Output_Error
      (CNN_Structure                   : in out CNN_Structure_Type;
       Gold_Output_Layer              : in     Real_Array) return Real;

    procedure Backward
      (Model                  : in out Model_Type;
       CNN_Structure          : access CNN_Structure_Type;
       Gradient               : in out Gradient_Type);

    procedure Decrease_Learning_Rate
      (Model : in out Model_Type;
       Epoch : in     Positive)
      with Inline;
    -- Learning Rate decreasing:
    -- (Initial_LR) / (1 + (LR_Decrease_onst * Epoch))

    procedure Initialize_CNN_Structure
      (CNN_Structure      : in out CNN_Structure_Type);

    procedure Finalize_CNN_Structure
      (CNN_Structure : in out CNN_Structure_Type);

    ----
    -- Errors
    ----

    CNN_Exception : exception;

--private

    type Model_Type is tagged record
        Configuration  : Configuration_Type;
        Is_Initialized : Boolean := False;

        Learning_Rate  : Real := 0.0;
        -- Learning rate (could change across epochs)

        -- 1D Filter Matrix used on CNN
        -- TODO: Multiple Filters
        W_In           : Real_Matrix_Access := null;
        B_In           : Real_Array_Access  := null;

        ---
        -- ADAM
        ---

        Wm_In          : Real_Matrix_Access := null;
        Bm_In          : Real_Array_Access  := null;
        Wv_In          : Real_Matrix_Access := null;
        Bv_In          : Real_Array_Access  := null;


        Timestep       : Real := 0.0;
        -- Timestep used for ADAM learning rate

    end record;
    -- Model/Classifier type

end MLColl.Neural_Networks.CNN;
