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

package body MLColl.Neural_Networks.Recurrent.RNN is
    
    package Stream_IO renames Ada.Streams.Stream_IO;

    procedure Initialize_RNN_Structure
      (RNN_Structure      : in out RNN_Structure_Type) is  
        
        Input_Layer_Size   : Positive renames RNN_Structure.Input_Layer_Size;
        Hidden_Layer_Size  : Positive renames RNN_Structure.Hidden_Layer_Size;
        Output_Layer_Size  : Positive renames RNN_Structure.Output_Layer_Size;
    begin
        
        RNN_Structure.Sequence_Hidden(RNN_Structure.First_Hidden_Sequence_Index) := new Real_Array
          (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);
            
        RNN_Structure.Sequence_Hidden_Derivative(RNN_Structure.First_Hidden_Sequence_Index) := new Real_Array
          (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);
        
        for T in RNN_Structure.First_Sequence_Index .. RNN_Structure.Last_Sequence_Index loop
            RNN_Structure.Sequence_Input(T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Input_Layer_Size) -  1);
            
            RNN_Structure.Sequence_Input_Gradients(T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Input_Layer_Size) -  1);
            
            RNN_Structure.Sequence_Hidden(T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);

            RNN_Structure.Sequence_Hidden_Derivative(T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);

            RNN_Structure.Sequence_Output(T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Output_Layer_Size) -  1);
                
            RNN_Structure.Sequence_Gold(T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Output_Layer_Size) -  1);
            
            RNN_Structure.Sequence_Output_Error(T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Output_Layer_Size) -  1);
        end loop;
          
    end Initialize_RNN_Structure;
    
    procedure Finalize_RNN_Structure
      (RNN_Structure : RNN_Structure_Type) is
    begin
        Free(RNN_Structure.Sequence_Input);
        Free(RNN_Structure.Sequence_Input_Gradients);
        Free(RNN_Structure.Sequence_Hidden);
        Free(RNN_Structure.Sequence_Hidden_Derivative);
        Free(RNN_Structure.Sequence_Output);
        Free(RNN_Structure.Sequence_Gold);
        Free(RNN_Structure.Sequence_Output_Error);
    end Finalize_RNN_Structure;
    
    procedure Forward
      (Model                   : in     Model_Type;
       RNN_Structure           : access RNN_Structure_Type) is separate;
    
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
        Text_IO.Put_Line (File, "Hidden_Layer_Size:"     & Model.Configuration.Hidden_Layer_Size'Img);
        Text_IO.Put_Line (File, "Initial_Learning_Rate:" & Model.Configuration.Initial_Learning_Rate'Img);
        Text_IO.Put_Line (File, "Random_Weights_Range:"  & Model.Configuration.Random_Weights_Range'Img);
        Text_IO.Put_Line (File, "Max_Num_Of_Epochs:"     & Model.Configuration.Max_Num_Of_Epochs'Img);
        Text_IO.Put_Line (File, "Min_Iterations:"        & Model.Configuration.Min_Num_Of_Epochs'Img);
        Text_IO.Put_Line (File, "Max_No_Best_Epochs:"    & Model.Configuration.Max_No_Best_Epochs'Img);
        Text_IO.Put_Line (File, "Learning_Rate_Decrease_Constant:" & Model.Configuration.Learning_Rate_Decrease_Constant'Img);
    end Print;
   
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
        Free (Model.W_Rec);
        
        --ADAM
        
        if Model.Configuration.Learning_Rule = ADAM then
            Model.Timestep := 0.0;
            
            Free (Model.Wm_In);
            Free (Model.Bm_In);
            Free (Model.Wm_Out);
            Free (Model.Bm_Out);
            Free (Model.Wv_In);
            Free (Model.Bv_In);
            Free (Model.Wv_Out);
            Free (Model.Bv_Out);
            Free (Model.Wv_Rec);
            Free (Model.Wm_Rec);
        end if;

        Model.Is_Initialized := False;        
    end Finalize;
    
    procedure Initialize_Matrices
      (Model : in out Model_Type) is separate;
    

    function Calculate_Binary_Output_Error
      (Model                : in     Model_Type;
       Actual_Output_Layer  : in     Real_Array_Access_Array;
       Gold_Output_Layer    : in     Real_Array_Access_Array;
       Output_Error         : in out Real_Array_Access_Array) return Real is
        pragma Unreferenced (Model);
        
    begin
        if Output_Error'Length /= Actual_Output_Layer'Length then
            raise Multilayer_Perceptron_Exception 
              with "Calculate_Binary_Output_Error_Sequence: Output_Error'Length /= Actual_Output_Layer'Length";
        end if;
        
        return Loss : Real := 0.0 do
        
            for T in Gold_Output_Layer'Range loop
                Output_Error(T) := new Real_Array(Actual_Output_Layer(T).all'Range);  
                
                declare
                    Gold_Outcome_Index : constant Index_Type
                      := Get_Max_Index(V => Gold_Output_Layer(T).all);
                begin
                    if Gold_Output_Layer(T)(Gold_Outcome_Index) /= 1.0 then
                        raise Multilayer_Perceptron_Exception
                          with "Calculate_Binary_Output_Error_Sequence: invalid Gold_Outcome_Index value";
                    end if;

                    -- Error
                    Output_Error(T)(Gold_Outcome_Index) := Output_Error(T)(Gold_Outcome_Index) - 1.0;
    
                    -- Loss
                    for I in Gold_Output_Layer(T)'Range loop
                        Loss := Loss + (0.5 * ((Gold_Output_Layer(T)(I) - Actual_Output_Layer(T)(I)) ** 2));
                    end loop;
                    Loss := Loss / Real(Gold_Output_Layer(T)'Length);
                end;
                
            end loop;
            
        end return;
        
    end Calculate_Binary_Output_Error;
    
    function Calculate_Output_Error
      (Model                : in     Model_Type;
       Actual_Output_Layer  : in     Real_Array_Access_Array;
       Gold_Output_Layer    : in     Real_Array_Access_Array;
       Output_Error         : in out Real_Array_Access_Array) return Real is
        pragma Unreferenced (Model);
    begin
        if Output_Error'Length /= Actual_Output_Layer'Length then
            raise Multilayer_Perceptron_Exception 
              with "Calc_Output_Error: Output_Error'Length /= Actual_Output_Layer'Length";
        end if;
        
        return Loss : Real := 0.0 do
        
            for T in Gold_Output_Layer'Range loop
                Output_Error(T) 
                  := new Real_Array(Actual_Output_Layer(T).all'Range);  
                
                Subtraction
                  (V_Out => Output_Error(T).all,
                   V1    => Actual_Output_Layer(T).all,
                   V2    => Gold_Output_Layer(T).all);
               
                for I in Gold_Output_Layer(T)'Range loop
                    Loss := Loss + (0.5 * ((Gold_Output_Layer(T)(I) - Actual_Output_Layer(T)(I)) ** 2));
                end loop;
                Loss := Loss / Real(Gold_Output_Layer(T)'Length);
            end loop;
            
        end return;
        
    end Calculate_Output_Error;
      
    procedure Backward
      (Model                  : in out Model_Type;
       RNN_Structure          : access RNN_Structure_Type) is separate;
   
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
        Gradient.B_Out := (others => 0.0);
        Gradient.W_Out := (others => (others => 0.0));
        Gradient.W_Rec := (others => (others => 0.0));
        Gradient.Input_Non_Zero := (others => False);
    end Reset_Gradients;
        
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
    
end MLColl.Neural_Networks.Recurrent.RNN;
