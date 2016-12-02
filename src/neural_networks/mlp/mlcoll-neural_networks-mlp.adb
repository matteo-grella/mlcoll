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

package body MLColl.Neural_Networks.MLP is
    
    package Stream_IO renames Ada.Streams.Stream_IO;

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
    
    procedure Print_NN_Structure_Size
      (NN_Structure : NN_Structure_Type) is
    begin
        Text_IO.Put_Line (NN_Structure.Input_Layer'First'Img  & " .. " & NN_Structure.Input_Layer'Last'Img);
        Text_IO.Put_Line (NN_Structure.Hidden_Layer'First'Img & " .. " & NN_Structure.Hidden_Layer'Last'Img);
        Text_IO.Put_Line (NN_Structure.Output_Layer'First'Img & " .. " & NN_Structure.Output_Layer'Last'Img);
    end Print_NN_Structure_Size;
    
    procedure Print
      (Model : in out Model_Type;
       File  : Text_IO.File_Type := Text_IO.Standard_Output) is
    begin
        Text_IO.Put_Line (File, "Hidden_Layer_Size:"     & Model.Configuration.Hidden_Layer_Size'Img);
        Text_IO.Put_Line (File, "Initial_Learning_Rate:" & Model.Configuration.Initial_Learning_Rate'Img);
        Text_IO.Put_Line (File, "Random_Weights_Range:"  & Model.Configuration.Random_Weights_Range'Img);
        Text_IO.Put_Line (File, "Max_Num_Of_Epochs:"     & Model.Configuration.Max_Num_Of_Epochs'Img);
        Text_IO.Put_Line (File, "Min_Iterations:"        & Model.Configuration.Min_Num_Of_Epochs'Img);
        Text_IO.Put_Line (File, "Max_No_Best_Epochs:"    & Model.Configuration.Max_No_Best_Epochs'Img);
        Text_IO.Put_Line (File, "Learning_Rate_Decrease_Constant:" & Model.Configuration.Learning_Rate_Decrease_Constant'Img);
    end Print;
    
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
    
    function Make_NN_Structure
      (Input_Layer_Size    : in     Positive;
       Hidden_Layer_Size   : in     Positive;
       Output_Layer_Size   : in     Positive) return NN_Structure_Type is
        
        RNN_Structure : NN_Structure_Type
          (Input_Layer_Size        => Input_Layer_Size,
           Hidden_Layer_Size       => Hidden_Layer_Size,
           Output_Layer_Size       => Output_Layer_Size,
           Input_Layer_Last        => Index_Type'First + Index_Type (Input_Layer_Size) - 1,
           Hidden_Layer_Last       => Index_Type'First + Index_Type(Hidden_Layer_Size) - 1,
           Output_Layer_Last       => Index_Type'First + Index_Type(Output_Layer_Size) - 1);
    begin
        return RNN_Structure;
    end Make_NN_Structure;
    
    procedure Initialize
      (Model                : in out Model_Type;
       Configuration        : in     Configuration_Type;
       Initialize_Weights   : in     Boolean := False) is
    begin
        
        if Model.Is_Initialized then
            raise Multilayer_Perceptron_Exception
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
            raise Multilayer_Perceptron_Exception
              with "Model not initialized";
        end if;
        
        Model.Learning_Rate := Model.Configuration.Initial_Learning_Rate;
        
        Free (Model.W_In);
        Free (Model.B_In);
        Free (Model.W_Out);
        Free (Model.B_Out);
        
        if Model.Configuration.Learning_Rule = ADAM then
            
            Free (Model.Wm_In);
            Free (Model.Bm_In);
            Free (Model.Wm_Out);
            Free (Model.Bm_Out);
            Free (Model.Wv_In);
            Free (Model.Bv_In);
            Free (Model.Wv_Out);
            Free (Model.Bv_Out);
            
            Model.Timestep := 0.0;
            
        elsif Model.Configuration.Learning_Rule = ADAGRAD then
            
            Free (Model.Wv_In);
            Free (Model.Bv_In);
            Free (Model.Wv_Out);
            Free (Model.Bv_Out);
            
        end if;
    
        Model.Is_Initialized := False;        
    end Finalize;
    
    procedure Initialize_Matrices
      (Model : in out Model_Type) is separate;
        
    function Calculate_Binary_Output_Error
      (Model                           : in     Model_Type;
       NN_Structure                    : in out NN_Structure_Type;
       Gold_Active_Outcome_Indexes     : in     Extended_Index_Array_Type;
       Constraint_Error                : in     Real := 0.0) return Real is
        
        Output_Error        : Real_Array renames NN_Structure.Output_Error;
        Normalized_Output   : Real_Array renames NN_Structure.Normalized_Output;        
        
        Constraint_Pi  : Real     renames Model.Configuration.Constraint_Hyperparams.Pi;
        Constraint_C   : Positive renames Model.Configuration.Constraint_Hyperparams.C;

        Constraint_Regularization : constant Real := 1.0 - Exp (- Real (Constraint_C) * Constraint_Error);
        
        Loss_Partition            : constant Real := 1.0 - Constraint_Pi;        
        Constraint_Partition      : constant Real := Constraint_Pi;

        Loss_Partition_Enabled : constant Boolean 
          := Constraint_Error /= 0.0 and then Constraint_Pi /= 0.0;
        
        Loss : Real := 0.0;
    begin
   
        Output_Error := Normalized_Output;
            
        for Best_Outcome_Index of Gold_Active_Outcome_Indexes loop

            -- Calculate best outcome loss 

            if Loss_Partition_Enabled then
                for J in Normalized_Output'Range loop
                    Output_Error (J) 
                      := (Loss_Partition * (Normalized_Output (J) - (if J = Best_Outcome_Index then 1.0 else 0.0)))
                      + (Constraint_Partition * Normalized_Output (J) * Constraint_Regularization);
                end loop;
            else    
                Output_Error (Best_Outcome_Index) := Normalized_Output (Best_Outcome_Index) - 1.0;
                -- Put_Line ("    [Best Outcome Error]");
                -- Put_Line ("        " & Output_Error (Best_Outcome_Index)'Img);
            end if;
                    
            -- Loss
            
            for J in Output_Error'Range loop
                declare
                    Loss_Contribute : constant Real 
                      := (((if J = Best_Outcome_Index then 1.0 else 0.0) - Normalized_Output (J)) ** 2);
                                
                    Constraint_Contribute : constant Real
                      := (Constraint_Regularization * Normalized_Output (J)) ** 2;
                begin
                    Loss := Loss + (0.5 * (Loss_Partition * Loss_Contribute + Constraint_Partition * Constraint_Contribute));
                end;
            end loop;
            
            Loss := Loss / Real (Output_Error'Length);
                    
        end loop;
        
        return Loss;
    end Calculate_Binary_Output_Error;
      
    function Calculate_Output_Error
      (Model                          : in     Model_Type;
       NN_Structure                   : in out NN_Structure_Type;
       Gold_Output_Layer              : in     Real_Array;
       Constraint_Error               : in     Real := 0.0) return Real is

        Output_Error        : Real_Array renames NN_Structure.Output_Error;
        Output              : Real_Array renames NN_Structure.Output_Layer;
       
        Constraint_Pi             : Real     renames Model.Configuration.Constraint_Hyperparams.Pi;
        Constraint_C              : Positive renames Model.Configuration.Constraint_Hyperparams.C;
        Constraint_Regularization : constant Real := 1.0 - Exp (- Real (Constraint_C) * Constraint_Error);        
        Loss_Partition            : constant Real := 1.0 - Constraint_Pi;        
        Constraint_Partition      : constant Real := Constraint_Pi;
        
        Loss_Partition_Enabled : constant Boolean 
          := Constraint_Error = 0.0 or else Constraint_Pi = 0.0;
        
        Loss : Real := 0.0;
    begin
        
        if Output_Error'Length /= Output'Length then
            raise Multilayer_Perceptron_Exception 
              with "Calc_Output_Error: Output_Error'Length /= Actual_Output_Layer'Length";
        end if;
        
        Output_Error := Output;
            
        -- Calculate loss for each outcome       

        for J in Output'First .. Output'Last loop
                        
            if Loss_Partition_Enabled then
                
                -- Calculate Error
                
                Output_Error (J) 
                  := (Loss_Partition * (Output (J) - Gold_Output_Layer (J)))
                  + (Constraint_Partition * Output (J) * Constraint_Regularization);
                    
                -- Calculate Loss
                
                declare
                    Loss_Contribute : constant Real 
                      := ((Gold_Output_Layer (J) - Output (J)) ** 2);
                                
                    Constraint_Contribute : constant Real
                      := (Constraint_Regularization * Output (J)) ** 2;
                begin
                    Loss := Loss + (0.5 * (Loss_Partition * Loss_Contribute + Constraint_Partition * Constraint_Contribute));
                end;

            else
                -- Calculate Error
                Output_Error (J) := Output (J) - Gold_Output_Layer (J);
                    
                -- Calculate Loss
                Loss := Loss + (0.5 * ((Gold_Output_Layer (J) - Output (J)) ** 2));        
            end if;                        
                        
        end loop;
                    
        Loss := Loss / Real (Output_Error'Length);
             
        return Loss;
    end Calculate_Output_Error;

    procedure Forward
      (Model         : in Model_Type;
       NN_Structure  : access NN_Structure_Type) is separate;
--       NN_Structure  : in out NN_Structure_Type) is separate;
    
    procedure Forward_With_Relevance
      (Model         : in Model_Type;
       NN_Structure  : in out NN_Structure_Type) is separate;
    
    procedure Backward
      (Model                : in out Model_Type;
       NN_Structure         : in out NN_Structure_Type;
       Gradient             : in out Gradient_Type;
       Accumulate_Gradients : in Boolean := False) is separate;
   
    procedure Weight_Update_SGD
      (Model            : in out Model_Type;
       Gradient         : in     Gradient_Type) is separate;
    
    procedure Weight_Update_ADAM
      (Model            : in out Model_Type;
       Gradient         : in     Gradient_Type) is separate;
    
    procedure Weight_Update_ADAGRAD
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
                Weight_Update_ADAGRAD
                  (Model                => Model,
                   Gradient             => Gradient);
        end case;
    end Weight_Update;

    procedure Reset_Gradients
      (Gradient : in out Gradient_Type) is
    begin            
        Gradient.B_In  := (others => 0.0);
        Gradient.W_In  := (others => (others => 0.0));
        Gradient.B_Out := (others => 0.0);
        Gradient.W_Out := (others => (others => 0.0));
        Gradient.Input_Non_Zero := (others => False);
        Gradient.Count := 1;
    end Reset_Gradients;
    
    procedure Reset_NN_Structure
      (NN_Structure : in out NN_Structure_Type) is
    begin        
        NN_Structure.Input_Layer            := (others => 0.0);
        NN_Structure.Hidden_Layer           := (others => 0.0);
        NN_Structure.Activated_Hidden_Layer := (others => 0.0);
        NN_Structure.Output_Layer           := (others => 0.0);
        NN_Structure.Normalized_Output      := (others => 0.0);
        NN_Structure.Output_Error           := (others => 0.0);
        NN_Structure.Input_Error            := (others => 0.0);
    end Reset_NN_Structure;
      
    procedure Add_L2_Regularization
      (Model       : in out Model_Type;
       Gradient    : in out Gradient_Type) is separate;
    
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
                
                --  if Model.Learning_Rate /= Model.Learning_Rate_Final and then Epoch > 1 then
                --    --Model.Learning_Rate := Exp((Model.Configuration.Max_Num_Of_Epochs - Epoch)); 
                --    --exp(((iterations - iteration) * log(trainer.learning_rate) + log(trainer.learning_rate_final)) / (iterations - iteration + 1));
                --  end if;
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
            raise Multilayer_Perceptron_Exception
              with "Loaded model is not initialized";
        end if;
        
    end Load;
    
end MLColl.Neural_Networks.MLP;
