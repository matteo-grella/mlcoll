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

package MLColl.Neural_Networks.Recurrent.RNN is

    type Configuration_Type is record
        
        Input_Layer_Size  : Positive;
        -- Number of Neuron in the Input Layer

        Hidden_Layer_Size : Positive := 200;
        -- Number of Neuron in the Input Layer

        Output_Layer_Size : Positive := 1;
        -- Number of Neuron in the Output Layer

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
        
        W_In : Real_Matrix
          (Index_Type'First .. Input_Layer_Last, 
           Index_Type'First .. Hidden_Layer_Last) 
          := (others => (others => 0.0));
        
        B_Out : Real_Array
          (Index_Type'First .. Output_Layer_Last)
          := (others => 0.0);
        
        W_Out : Real_Matrix
          (Index_Type'First .. Hidden_Layer_Last, 
           Index_Type'First .. Output_Layer_Last) 
          := (others => (others => 0.0));

        W_Rec : Real_Matrix
          (Index_Type'First .. Hidden_Layer_Last,
           Index_Type'First .. Hidden_Layer_Last) 
          := (others => (others => 0.0));

        Input_Non_Zero : Boolean_Array
          (Index_Type'First .. Input_Layer_Last) 
          := (others => False);
    end record;
    
    type RNN_Structure_Type
      (First_Sequence_Index        : Index_Type;
       Last_Sequence_Index         : Index_Type;
       First_Hidden_Sequence_Index : Extended_Index_Type;
       Input_Layer_Size            : Positive;
       Hidden_Layer_Size           : Positive;
       Output_Layer_Size           : Positive;
       Input_Layer_Last            : Index_Type;
       Hidden_Layer_Last           : Index_Type;
       Output_Layer_Last           : Index_Type) is record

        Sequence_Input             : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Input_Gradients   : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Hidden            : RNN_Array_Of_Float_Vectors_Type (First_Hidden_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Hidden_Derivative : RNN_Array_Of_Float_Vectors_Type (First_Hidden_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Output            : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Gold              : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Output_Error      : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        
        Gradient                   : Gradient_Type
          (Input_Layer_Last   => Input_Layer_Last,
           Hidden_Layer_Last  => Hidden_Layer_Last,
           Output_Layer_Last  => Output_Layer_Last);
    end record;

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
       RNN_Structure           : access RNN_Structure_Type);
    
    procedure Reset_Gradients
      (Gradient : in out Gradient_Type);
    
    procedure Add_L2_Regularization
      (Model       : in out Model_Type;
       Gradient    : in out Gradient_Type);

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
      (Model                : in     Model_Type;
       Actual_Output_Layer  : in     Real_Array_Access_Array;
       Gold_Output_Layer    : in     Real_Array_Access_Array;
       Output_Error         : in out Real_Array_Access_Array) return Real;
    -- optimize loss calculation for binary outcomes
    
    function Calculate_Binary_Output_Error
      (Model                : in     Model_Type;
       Actual_Output_Layer  : in     Real_Array_Access_Array;
       Gold_Output_Layer    : in     Real_Array_Access_Array;
       Output_Error         : in out Real_Array_Access_Array) return Real;
    
    procedure Backward
      (Model                  : in out Model_Type;
       RNN_Structure          : access RNN_Structure_Type);
    -- 

    procedure Decrease_Learning_Rate
      (Model : in out Model_Type;
       Epoch : in     Positive)
      with Inline;
    -- Learning Rate decreasing:
    -- (Initial_LR) / (1 + (LR_Decrease_onst * Epoch))

    procedure Initialize_RNN_Structure
      (RNN_Structure      : in out RNN_Structure_Type);

    procedure Finalize_RNN_Structure
      (RNN_Structure : RNN_Structure_Type);
    
    ----
    -- Errors
    ----

    Multilayer_Perceptron_Exception : exception;

--private
        
    type Model_Type is tagged record
        Configuration  : Configuration_Type;
        Is_Initialized : Boolean := False;
        
        Learning_Rate  : Real := 0.0;
        -- Learning rate (could change across epochs)

        W_In           : Real_Matrix_Access := null;
        B_In           : Real_Array_Access  := null;
        W_Out          : Real_Matrix_Access := null;
        B_Out          : Real_Array_Access  := null;
        W_Rec          : Real_Matrix_Access := null;
        
        ---
        -- ADAM
        ---

        Wm_In          : Real_Matrix_Access := null;
        Bm_In          : Real_Array_Access  := null;
        Wv_In          : Real_Matrix_Access := null;
        Bv_In          : Real_Array_Access  := null;

        Wm_Out         : Real_Matrix_Access := null;
        Bm_Out         : Real_Array_Access  := null;
        Wv_Out         : Real_Matrix_Access := null;
        Bv_Out         : Real_Array_Access  := null;

        Wv_Rec         : Real_Matrix_Access := null;
        Wm_Rec         : Real_Matrix_Access := null;
        
        Timestep       : Real := 0.0;
        -- Timestep used for ADAM learning rate
        
    end record;
    -- Model/Classifier type
    
end MLColl.Neural_Networks.Recurrent.RNN;
