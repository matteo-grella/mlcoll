--
--  Copyright 2013 M. Grella, S. Cangialosi. All Rights Reserved.
--
--  Licensed under the Apache License, Version 2.0 (the "License");
--  you may not use this file except in compliance with the License.
--  You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
--  Unless required by applicable law or agreed to in writing, software
--  distributed under the License is distributed on an "AS IS" BASIS,
--  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
--  See the License for the specific language governing permissions and
--  limitations under the License.
--
---------------------------------------------------------------------------

pragma License (GPL);

with Text_IO; use Text_IO;
with Ada.Float_Text_IO;
with Ada.Command_Line;
with Ada.Directories;

with ARColl; use ARColl;
with ARColl.IO; use ARColl.IO;
with ARColl.Numerics.Reals; use ARColl.Numerics.Reals;
with ARColl.Numerics.Reals.Functions; use ARColl.Numerics.Reals.Functions;
with ARColl.Progress;

with MLColl.Neural_Networks; use MLColl.Neural_Networks;
with MLColl.Neural_Networks.MLP; use MLColl.Neural_Networks.MLP;
with MLColl.Neural_Networks.Datasets; use MLColl.Neural_Networks.Datasets;
with MLColl.Neural_Networks.Datasets.JSON_Loader;

procedure Test_MNIST is

    procedure Initialize_Network
      (Network_Model     : in out MLColl.Neural_Networks.MLP.Model_Type;
       Input_Layer_Size  : in     Positive;
       Hidden_Layer_Size : in     Positive;
       Output_Layer_Size : in     Positive) is
    begin
        
        Initialize
          (Model              => Network_Model,
           Configuration      =>
             (Input_Layer_Size                => Input_Layer_Size,
              Hidden_Layer_Size               => Hidden_Layer_Size,
              Output_Layer_Size               => Output_Layer_Size,
              Features_Type                   => DENSE,
              Initial_Learning_Rate           => 0.01,
              Random_Weights_Range            => 0.1,
              Max_Num_Of_Epochs               => 0,
              Min_Num_Of_Epochs               => 0,
              Max_No_Best_Epochs              => 0,
              Learning_Rate_Decrease_Constant => 0.5,
              Activation_Function_Name        => ReLU,
              Leaky_ReLU_Factor               => 0.01,
              Learning_Rule                   => SGD,
              Regularization_Parameter        => 1.0E-8,
              ADAM_Hypermarams                => Default_ADAM_Hyperparams,
              Constraint_Hyperparams          => 
                (Pi => 0.1, -- 0.1 standard parameter
                 C  => 10)),
           Initialize_Weights => True);

        New_Line (Standard_Error);
        Put_Line (Standard_Error, "[NN]     Learning rate: " & Network_Model.Learning_Rate'Img);
        Put_Line (Standard_Error, "[NN]  Input_Layer_Size: " & Get_Input_Layer_Size (Network_Model)'Img);
        Put_Line (Standard_Error, "[NN] Hidden_Layer_Size: " & Get_Hidden_Layer_Size (Network_Model)'Img);
        Put_Line (Standard_Error, "[NN] Output_Layer_Size: " & Get_Output_Layer_Size (Network_Model)'Img);
    
    end Initialize_Network;

    function Learn
      (Network_Model     : in out MLColl.Neural_Networks.MLP.Model_Type;
       Example           : in     Example_Type) return Real is
        
        Loss : Real := 0.0;
    
        NN_Structure : aliased NN_Structure_Type 
          := Make_NN_Structure 
            (Input_Layer_Size  => Network_Model.Configuration.Input_Layer_Size,
             Hidden_Layer_Size => Network_Model.Configuration.Hidden_Layer_Size,
             Output_Layer_Size => Network_Model.Configuration.Output_Layer_Size);
        
        Gradient  : Gradient_Type
          := Make_Gradient
            (Input_Layer_Size  => NN_Structure.Input_Layer_Size,
             Hidden_Layer_Size => NN_Structure.Hidden_Layer_Size,
             Output_Layer_Size => NN_Structure.Output_Layer_Size);
        
        Gold_Outcome_Indexes : constant Extended_Index_Array_Type 
          := (Index_Type'First => Example.Outcome_Index);
    begin
         

        NN_Structure.Input_Layer := Example.Features.all;
        
        Forward 
          (Model        => Network_Model,
           NN_Structure => NN_Structure'Access);

        Softmax
          (V              => NN_Structure.Output_Layer,
           SoftMax_Vector => NN_Structure.Normalized_Output);
                
        Loss := Calculate_Binary_Output_Error 
          (Model                           => Network_Model,
           NN_Structure                    => NN_Structure,
           Gold_Active_Outcome_Indexes     => Gold_Outcome_Indexes,
           Constraint_Error                => 0.0);

        Backward 
          (Model          => Network_Model,
           NN_Structure   => NN_Structure,
           Gradient       => Gradient);
            
        Weight_Update
          (Model       => Network_Model,
           Gradient    => Gradient);
        
        return Loss;
        
    end Learn;
    
    procedure Print_Image 
      (Image_Array  : Real_Array;
       Is_Relevance : Boolean := False) is
        
        Threshold : constant Real := (if Is_Relevance then 0.0 else 0.5);
    begin
        
        for I in Image_Array'Range loop
            if I > 0 and I mod 28 = 0 then
                New_Line;
            end if;
            
            if Image_Array (I) < Threshold then
                Put ("  ");
            else
                Put (" #");
            end if; 
        end loop;
        
        New_Line;
        
    end Print_Image;
    
    function Evaluate
      (Network_Model     : in MLColl.Neural_Networks.MLP.Model_Type;
       Example           : in Example_Type;
       Print_Relevance   : in Boolean := False) return Boolean is
        
        function Get_Output 
          (Output_Layer : Real_Array) return Natural is
            
            Best_Outcome       : Real       := Output_Layer ( Index_Type'First );
            Best_Outcome_Index : Index_Type := Index_Type'First;
        begin
            
            for I in Index_Type'First + 1 .. Output_Layer'Last loop
                if Output_Layer (I) > Best_Outcome then
                    Best_Outcome := Output_Layer (I);
                    Best_Outcome_Index := I;
                end if;
            end loop;
            
            return Natural (Best_Outcome_Index - Index_Type'First);
        end;

        NN_Structure : aliased NN_Structure_Type 
          := Make_NN_Structure 
            (Input_Layer_Size  => Network_Model.Configuration.Input_Layer_Size,
             Hidden_Layer_Size => Network_Model.Configuration.Hidden_Layer_Size,
             Output_Layer_Size => Network_Model.Configuration.Output_Layer_Size);
        
        Arg_Max_Index : Extended_Index_Type := -1;
        
    begin
        
        NN_Structure.Input_Layer := Example.Features.all;
                
        If Print_Relevance then
            Forward_With_Relevance
              (Model        => Network_Model,
               NN_Structure => NN_Structure);
                        
            Put_Line ("Features");
            Print_Image (NN_Structure.Input_Layer);
            Put_Line ("Output: " & Get_Output (NN_Structure.Output_Layer)'Img);
            Put_Line ("Relevance");
            Print_Image (NN_Structure.Input_Relevance, Is_Relevance => True);
        else
            Forward
              (Model        => Network_Model,
               NN_Structure => NN_Structure'Access);
        end if;

        Arg_Max_Index := Softmax
          (V              => NN_Structure.Output_Layer,
           SoftMax_Vector => NN_Structure.Normalized_Output);
        
        return (Arg_Max_Index = Example.Outcome_Index);            
            
    end Evaluate;

    function Validate
      (Network_Model   : in MLColl.Neural_Networks.MLP.Model_Type;
       Dataset         : in Dataset_Type;
       Print_Relevance : in Boolean := False) return Natural is
        use ARColl.Progress;
        
        Correct_Predictions : Natural := 0;
        
        Indicator : Progress_Indicator := Create_Indicator
          (Total => Natural (Dataset.Length), Mode => BAR);
        
    begin
        
        for Example of Dataset loop
            declare
                Prediction_Is_Correct : Boolean;
            begin
                Prediction_Is_Correct := Evaluate
                  (Network_Model   => Network_Model,
                   Example         => Example,
                   Print_Relevance => Print_Relevance);
                
                if Prediction_Is_Correct then
                    Correct_Predictions := Correct_Predictions + 1;
                end if;
                
                Indicator.Tick;
            end;
        end loop;
        
        return Correct_Predictions;
        
    end Validate;
    
    procedure Train
      (Network_Model     : in out MLColl.Neural_Networks.MLP.Model_Type;
       Training_Set      : in     Dataset_Type;
       Validation_Set    : in     Dataset_Type;
       Max_Iter          : in     Positive) is
        
        Random_Indexes : Index_Type_Array
          (Training_Set.First_Index .. Training_Set.Last_Index);
        
        use ARColl.Progress;

        Indicator : Progress_Indicator := Create_Indicator
          (Total => Natural (Training_Set.Length), Mode => BAR);
        
    begin

        Iter_Loop : for Iter in 1 .. Max_Iter loop

            New_Line (Standard_Error);
            Put_Line (Standard_Error, "Iteration " & Img (Iter));
                        
            if Iter > 1 then
                Decrease_Learning_Rate (Network_Model, Iter);
            end if;
            
            Array_Random_Permutation
              (Arr       => Random_Indexes,
               Randomize => True);
            
            Indicator.Reset;
            
            Example_Loop : for Example_Index of Random_Indexes loop
                declare
                    Loss : Real := 0.0;
                    pragma Unreferenced (Loss);
                begin
                    Loss := Learn
                      (Network_Model => Network_Model,
                       Example       => Training_Set.Element (Example_Index));
                end;
                
                Indicator.Tick;
            end loop Example_Loop;
            
            -- TODO: if best..
            Serialize 
              (Model          => Network_Model,
               Model_Filename => "model");
    
            Put_Line ("Validation");
    
            declare
                Correct_Predictions : Natural;
            begin
                Correct_Predictions := Validate
                  (Network_Model => Network_Model,
                   Dataset       => Validation_Set);
        
                Put ("Accuracy: ");
                Ada.Float_Text_IO.Put ( Float ( 100 * Correct_Predictions ) / Float (Validation_Set.Length), Aft => 2, Exp => 0 );
                Put_Line ("%");
            end;       
              
        end loop Iter_Loop;
        
    end Train;
    
    Program_Path : constant String := Ada.Command_Line.Command_Name;
    Program_Dir  : constant String := Ada.Directories.Containing_Directory (Program_Path);
    Data_Dir     : constant String := Path_Join (Path_Join (Program_Dir, ".."), "data");
    
    -----
    
    Test_Set       : Dataset_Type;
    Training_Set   : Dataset_Type;
    Validation_Set : Dataset_Type;
    
    ---
        
    Network_Model : MLColl.Neural_Networks.MLP.Model_Type;
    
begin
    
    Put_Line ("Test MNIST - Begin");
    
    -- Load datasets from JSON
    
    Test_Set       := JSON_Loader.Load_Dataset (Path_Join (Data_Dir, "mnist_test_set.json"));
    Training_Set   := JSON_Loader.Load_Dataset (Path_Join (Data_Dir, "mnist_training_set.json"));
    Validation_Set := JSON_Loader.Load_Dataset (Path_Join (Data_Dir, "mnist_validation_set.json"));

    pragma Assert (not Training_Set.Is_Empty);
    pragma Assert (not Test_Set.Is_Empty);
    pragma Assert (not Validation_Set.Is_Empty);
      
    Put_Line ("Initialize Network");
    
    Initialize_Network 
      (Network_Model     => Network_Model,
       Input_Layer_Size  => Training_Set.First_Element.Features'Length,
       Hidden_Layer_Size => 100,
       Output_Layer_Size => 10);
      
    Put_Line ("Training");
    
    Train
      (Network_Model  => Network_Model,
       Training_Set   => Training_Set,
       Validation_Set => Validation_Set,
       Max_Iter       => 10);
    
    Put_Line ("Final Evaluation");
    
    declare
        Correct_Predictions : Natural;
    begin
        
        Correct_Predictions := Validate
          (Network_Model   => Network_Model,
           Dataset         => Test_Set,
           Print_Relevance => True);
        
        Put ("Accuracy: ");
        Ada.Float_Text_IO.Put ( Float ( 100 * Correct_Predictions ) / Float (Validation_Set.Length), Aft => 2, Exp => 0 );
        Put_Line ("%");
    end;         
    
    Put_Line ("Test MNIST - End");
    
end Test_MNIST;
