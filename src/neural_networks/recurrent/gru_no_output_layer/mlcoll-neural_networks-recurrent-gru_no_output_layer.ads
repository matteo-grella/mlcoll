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

package MLColl.Neural_Networks.Recurrent.GRU_No_Output_Layer is
    
    type Configuration_Type is record
        Input_Layer_Size  : Positive;
        -- Number of Neuron in the Input Layer

        Hidden_Layer_Size : Positive := 100;
        -- Number of Neuron in the Input Layer

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
        -- Candidate (c), Reset (r), Interpolate (z) Gates
        ---

        Wc_In         : Real_Matrix_Access := null;
        Wr_In         : Real_Matrix_Access := null;
        Wz_In         : Real_Matrix_Access := null;

        Wc_Rec        : Real_Matrix_Access := null;
        Wr_Rec        : Real_Matrix_Access := null;
        Wz_Rec        : Real_Matrix_Access := null;

        Bc_In         : Real_Array_Access  := null;
        Br_In         : Real_Array_Access  := null;
        Bz_In         : Real_Array_Access  := null;

        --W_Out         : Real_Matrix_Access := null;
        --B_Out         : Real_Array_Access  := null;

        Learning_Rate : Real := 0.0;
        -- Learning rate (could change across epochs)

        ---
        -- ADAM
        ---

        Wcm_In         : Real_Matrix_Access := null;
        Bcm_In         : Real_Array_Access  := null;
        Wcv_In         : Real_Matrix_Access := null;
        Bcv_In         : Real_Array_Access  := null;
        Wcv_Rec        : Real_Matrix_Access := null;
        Wcm_Rec        : Real_Matrix_Access := null;
        Wzm_In         : Real_Matrix_Access := null;
        Bzm_In         : Real_Array_Access  := null;
        Wzv_In         : Real_Matrix_Access := null;
        Bzv_In         : Real_Array_Access  := null;
        Wzv_Rec        : Real_Matrix_Access := null;
        Wzm_Rec        : Real_Matrix_Access := null;
        Wrm_In         : Real_Matrix_Access := null;
        Brm_In         : Real_Array_Access  := null;
        Wrv_In         : Real_Matrix_Access := null;
        Brv_In         : Real_Array_Access  := null;
        Wrv_Rec        : Real_Matrix_Access := null;
        Wrm_Rec        : Real_Matrix_Access := null;

        Timestep        : Real := 0.0;

        Is_Initialized : Boolean := False;
    end record;
    -- Model/Classifier type

    type Gradient_Type
      (Input_Layer_Last  : Index_Type;
       Hidden_Layer_Last : Index_Type) is record
      
        Wr_In  : Real_Matrix
          (Index_Type'First .. Input_Layer_Last,
           Index_Type'First .. Hidden_Layer_Last);
      
        Wz_In  : Real_Matrix
          (Index_Type'First .. Input_Layer_Last,
           Index_Type'First .. Hidden_Layer_Last);
      
        Wc_In  : Real_Matrix
          (Index_Type'First .. Input_Layer_Last,
           Index_Type'First .. Hidden_Layer_Last);
      
        Wr_Rec : Real_Matrix
          (Index_Type'First .. Hidden_Layer_Last,
           Index_Type'First .. Hidden_Layer_Last);
      
        Wz_Rec : Real_Matrix
          (Index_Type'First .. Hidden_Layer_Last,
           Index_Type'First .. Hidden_Layer_Last);
      
        Wc_Rec : Real_Matrix
          (Index_Type'First .. Hidden_Layer_Last,
           Index_Type'First .. Hidden_Layer_Last);

        Br_In : Real_Array
          (Index_Type'First .. Hidden_Layer_Last);
      
        Bz_In : Real_Array
          (Index_Type'First .. Hidden_Layer_Last);
      
        Bc_In : Real_Array
          (Index_Type'First .. Hidden_Layer_Last);
        
        Count : Positive := 1;
        
    end record;
    
    type GRU_Structure_Type
      (First_Sequence_Index        : Index_Type;
       Last_Sequence_Index         : Index_Type;
       First_Hidden_Sequence_Index : Extended_Index_Type;
       Input_Layer_Size            : Positive;
       Hidden_Layer_Size           : Positive;
       Input_Layer_Last            : Index_Type;
       Hidden_Layer_Last           : Index_Type) is record

        Sequence_Input             : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Input_Gradients   : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);

        Sequence_Reset_Activations         : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Interpolate_Activations   : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Candidate_Activations     : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);

        Sequence_Hidden            : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        Sequence_Hidden_Derivative : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);

        Sequence_Hidden_Error      : Real_Array_Access_Array (First_Sequence_Index .. Last_Sequence_Index) := (others => null);
        
        Gradient                   : Gradient_Type
          (Input_Layer_Last  => Input_Layer_Last,
           Hidden_Layer_Last => Hidden_Layer_Last);
    end record;

    ----
    -- Functions
    ----

    procedure Initialize_GRU_Structure
      (GRU_Structure      : in out GRU_Structure_Type);

    procedure Finalize_GRU_Structure
      (GRU_Structure : GRU_Structure_Type);

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
       GRU_Structure           : access GRU_Structure_Type);
    --   GRU_Structure           : in out GRU_Structure_Type);
    -- Out: Hidden_Layers, Hidden_Layers_Deriv,

    procedure Backward
      (Model                     : in out Model_Type;
       GRU_Structure             : in out GRU_Structure_Type);
    
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

    GRU_Exception : exception;
    
end MLColl.Neural_Networks.Recurrent.GRU_No_Output_Layer;

