------------------------------------------------------------------------------
--                               M L C O L L
--  M a c h i n e   L e a r n i n g   C o m p o n e n t   C o l l e c t i o n
--
--        Copyright 2009-2016 M. Grella, S. Cangialosi, E. Brambilla
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

package MLColl.Neural_Networks.Recurrent.CFN is
    
    type Configuration_Type is record
        Input_Layer_Size  : Positive;
        -- Number of Neuron in the Input Layer

        Hidden_Layer_Size : Positive := 100;
        -- Number of Neuron in the Input Layer

        Output_Layer_Size : Positive := 1;
        -- Number of Neuron in the Output Layer

        Initial_Learning_Rate : Real;
        -- Initial learning rate

        Random_Weights_Range  : Real := 0.1;
        -- See "Get_Random_Weight" function defined
        -- inside "Initialize_Matrices" procedure

        Max_Num_Of_Epochs     : Integer := -1;
        -- Maximum number of training Epochs

        Min_Num_Of_Epochs     : Integer := -1;
        -- Minimum number of training Epochs

        Max_No_Best_Epochs    : Integer := -1;
        -- Maximum number of Epochs without a new "best" validation
        -- before train termination

        Activation_Function_Name : Activation_Function_Name_Type;

        Leaky_ReLU_Factor : Real := 0.01;

        Learning_Rule     : Learning_Rule_Type := ADAM;

        Propagate_Error_To_Input_Layer : Boolean := False;
        
        ADAM_Hypermarams               : ADAM_Hyperparams_Type;
    end record;

    type Model_Type is record
        Configuration : Configuration_Type;

        ---
        -- Candidate (c), Forget (f), Input (i) Gates
        ---

        Wc_In         : Real_Matrix_Access := null;
        Wf_In         : Real_Matrix_Access := null;
        Wi_In         : Real_Matrix_Access := null;

        --Wc_Rec        : Real_Matrix_Access := null;
        Wf_Rec        : Real_Matrix_Access := null;
        Wi_Rec        : Real_Matrix_Access := null;

        --Bc_In         : Real_Array_Access  := null;
        Bf_In         : Real_Array_Access  := null;
        Bi_In         : Real_Array_Access  := null;

        W_Out         : Real_Matrix_Access := null;
        B_Out         : Real_Array_Access  := null;
        

        Learning_Rate : Real := 0.0;
        -- Learning rate (could change across epochs)

        ---
        -- ADAM
        ---

        Wcm_In         : Real_Matrix_Access := null;
        Wcv_In         : Real_Matrix_Access := null;
        Wim_In         : Real_Matrix_Access := null;
        Bim_In         : Real_Array_Access  := null;
        Wiv_In         : Real_Matrix_Access := null;
        Biv_In         : Real_Array_Access  := null;
        Wiv_Rec        : Real_Matrix_Access := null;
        Wim_Rec        : Real_Matrix_Access := null;
        
        Wfm_In         : Real_Matrix_Access := null;
        Bfm_In         : Real_Array_Access  := null;
        Wfv_In         : Real_Matrix_Access := null;
        Bfv_In         : Real_Array_Access  := null;
        Wfv_Rec        : Real_Matrix_Access := null;
        Wfm_Rec        : Real_Matrix_Access := null;

        Wm_Out : Real_Matrix_Access := null;
        Bm_Out : Real_Array_Access  := null;
        Wv_Out : Real_Matrix_Access := null;
        Bv_Out : Real_Array_Access  := null;
        
        Timestep        : Real := 0.0;

        Is_Initialized : Boolean := False;
    end record;
    -- Model/Classifier type

    type Gradient_Type
      (Input_Layer_Last  : Index_Type;
       Hidden_Layer_Last : Index_Type;
       Output_Layer_Last : Index_Type) is record

        B_Out : Real_Array
          (Index_Type'First .. Output_Layer_Last);

        W_Out : Real_Matrix
          (Index_Type'First .. Hidden_Layer_Last,
           Index_Type'First .. Output_Layer_Last);
      
        Wf_In  : Real_Matrix
          (Index_Type'First .. Input_Layer_Last,
           Index_Type'First .. Hidden_Layer_Last);
      
        Wi_In  : Real_Matrix
          (Index_Type'First .. Input_Layer_Last,
           Index_Type'First .. Hidden_Layer_Last);
      
        Wc_In  : Real_Matrix
          (Index_Type'First .. Input_Layer_Last,
           Index_Type'First .. Hidden_Layer_Last);
      
        Wf_Rec : Real_Matrix
          (Index_Type'First .. Hidden_Layer_Last,
           Index_Type'First .. Hidden_Layer_Last);
      
        Wi_Rec : Real_Matrix
          (Index_Type'First .. Hidden_Layer_Last,
           Index_Type'First .. Hidden_Layer_Last);
      
        Bf_In : Real_Array
          (Index_Type'First .. Hidden_Layer_Last);
      
        Bi_In : Real_Array
          (Index_Type'First .. Hidden_Layer_Last);  
    end record;
    
    type CFN_Structure_Type
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

        Sequence_Forget_Activations         : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Input_Activations          : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Candidate_Activations      : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Hidden_Activations         : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);

        Sequence_Hidden            : RNN_Array_Of_Float_Vectors_Type (First_Hidden_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Hidden_Derivative : RNN_Array_Of_Float_Vectors_Type (First_Hidden_Sequence_Index .. Last_Sequence_Index) := (others => null);

        Sequence_Output            : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Gold              : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        
        Sequence_Output_Error      : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
               
        Gradient                   : Gradient_Type
          (Input_Layer_Last  => Input_Layer_Last,
           Hidden_Layer_Last => Hidden_Layer_Last,
           Output_Layer_Last => Output_Layer_Last);
    end record;

    ----
    -- Functions
    ----

    procedure Initialize_CFN_Structure
      (CFN_Structure      : in out CFN_Structure_Type);

    procedure Finalize_CFN_Structure
      (CFN_Structure : CFN_Structure_Type);

    procedure Initialize
      (Model              : in out Model_Type;
       Configuration      : in     Configuration_Type;
       Initialize_Weights : in     Boolean := False) with
      Pre => Configuration.Activation_Function_Name = Tanh;
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
      (Model               : in Model_Type) return Index_Type is
      (Index_Type'First + Index_Type (Model.Configuration.Output_Layer_Size) - 1) with Inline;

    function Get_Output_Layer_Size
      (Model  : in Model_Type) return Positive is
      (Model.Configuration.Output_Layer_Size) with Inline;

    function Get_Input_Layer_Size
      (Model  : in Model_Type) return Positive is
      (Model.Configuration.Input_Layer_Size) with Inline;

    function Get_Hidden_Layer_Size
      (Model  : in Model_Type) return Positive is
      (Model.Configuration.Hidden_Layer_Size) with Inline;

    function Get_Max_No_Best_Epochs
      (Model  : in Model_Type) return Positive is (Model.Configuration.Max_No_Best_Epochs) with Inline;

    function Get_Min_Num_Of_Epochs
      (Model  : in Model_Type) return Positive is (Model.Configuration.Min_Num_Of_Epochs) with Inline;

    ----
    -- Network Functions
    ----

    procedure Initialize_Matrices
      (Model : in out Model_Type);
    -- Matrices initialization

    procedure Forward
      (Model                   : in Model_Type;
       CFN_Structure           : access CFN_Structure_Type);
    --   GRU_Structure           : in out GRU_Structure_Type);
    -- Out: Hidden_Layers, Hidden_Layers_Deriv, Output_Layers

    function Calculate_Binary_Output_Error
      (Model                : in     Model_Type;
       Actual_Output_Layer  : in     Real_Array_Access_Array;
       Gold_Output_Layer    : in     Real_Array_Access_Array;
       Output_Error         : in out Real_Array_Access_Array) return Real;
          
    function Calculate_Output_Error
      (Model                : in     Model_Type;
       Actual_Output_Layer  : in     Real_Array_Access_Array;
       Gold_Output_Layer    : in     Real_Array_Access_Array;
       Output_Error         : in out Real_Array_Access_Array) return Real;

    procedure Backward
      (Model                     : in out Model_Type;
       CFN_Structure             : in out CFN_Structure_Type);
    
    procedure Weight_Update_SGD
      (Model       : in out Model_Type;
       Gradient    : in     Gradient_Type);

    procedure Weight_Update_ADAM
      (Model       : in out Model_Type;
       Gradient    : in     Gradient_Type);

    procedure Weight_Update
      (Model       : in out Model_Type;
       Gradient    : in     Gradient_Type);
    
    ----
    -- Errors
    ----

    CFN_Exception : exception;
    
end MLColl.Neural_Networks.Recurrent.CFN;
