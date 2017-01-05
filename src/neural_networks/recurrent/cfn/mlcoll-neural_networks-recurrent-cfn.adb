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

private with Ada.Streams.Stream_IO;

with ARColl.Numerics.Reals.Functions; use ARColl.Numerics.Reals.Functions;

package body MLColl.Neural_Networks.Recurrent.CFN is
    
    package Stream_IO renames Ada.Streams.Stream_IO;
    
    procedure Initialize
      (Model                : in out Model_Type;
       Configuration        : in     Configuration_Type;
       Initialize_Weights   : in     Boolean := False) is
    begin
        
        if Model.Is_Initialized then
            raise CFN_Exception
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
            raise CFN_Exception
              with "Model not initialized";
        end if;
        
        Free (Model.Wc_In);
        Free (Model.Wf_In);
        Free (Model.Wi_In);
        
        Free (Model.Wf_Rec);
        Free (Model.Wi_Rec);
        
        Free (Model.Bf_In);
        Free (Model.Bi_In);
        Free (Model.W_Out);
        Free (Model.B_Out);
        
        -- ADAM
        
        if Model.Configuration.Learning_Rule = ADAM then
            Free (Model.Wcm_In);
            Free (Model.Wcv_In);
            
         
            Free (Model.Wim_In);
            Free (Model.Bim_In);
            Free (Model.Wim_Rec);
            Free (Model.Wiv_In);
            Free (Model.Biv_In);
            Free (Model.Wiv_Rec);
         
            Free (Model.Wfm_In);
            Free (Model.Bfm_In); 
            Free (Model.Wfm_Rec);
            Free (Model.Wfv_In);
            Free (Model.Bfv_In);
            Free (Model.Wfv_Rec);

            Free (Model.Wm_Out);
            Free (Model.Bm_Out);
            Free (Model.Wv_Out);
            Free (Model.Bv_Out);
            
            Model.Timestep := 0.0;
        end if;
        
        Model.Learning_Rate := Model.Configuration.Initial_Learning_Rate;
        
        Model.Is_Initialized := False;        
    end Finalize;
    
    procedure Initialize_Matrices
      (Model : in out Model_Type) is separate;
          
    procedure Forward
      (Model                   :     in Model_Type;
       CFN_Structure           : access CFN_Structure_Type) is separate;
    
    function Calculate_Binary_Output_Error
      (Model                : in     Model_Type;
       Actual_Output_Layer  : in     Real_Array_Access_Array;
       Gold_Output_Layer    : in     Real_Array_Access_Array;
       Output_Error         : in out Real_Array_Access_Array) return Real is
        pragma Unreferenced (Model);
    begin
        
        if Output_Error'Length /= Actual_Output_Layer'Length then
            raise CFN_Exception 
              with "Calculate_Binary_Output_Error_Sequence: Output_Error'Length /= Actual_Output_Layer'Length";
        end if;
        
        return Loss : Real := 0.0 do
        
            for T in Gold_Output_Layer'Range loop
                Output_Error (T) 
                  := new Real_Array (Actual_Output_Layer (T).all'Range);  
                
                declare
                    Gold_Outcome_Index : constant Index_Type
                      := Get_Max_Index (V => Gold_Output_Layer (T).all);
                begin
                    if Gold_Output_Layer (T) (Gold_Outcome_Index) /= 1.0 then
                        raise CFN_Exception
                          with "Calculate_Binary_Output_Error_Sequence: invalid Gold_Outcome_Index value";
                    end if;

                    -- Error
                    Output_Error (T) (Gold_Outcome_Index) := Output_Error (T) (Gold_Outcome_Index) - 1.0;
    
                    -- Loss
                    for I in Gold_Output_Layer (T)'Range loop
                        Loss := Loss + (0.5 * ((Gold_Output_Layer (T) (I) - Actual_Output_Layer (T) (I)) ** 2));
                    end loop;
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
            raise CFN_Exception 
              with "Calc_Output_Error: Output_Error'Length /= Actual_Output_Layer'Length";
        end if;
        
        return Loss : Real := 0.0 do
        
            for T in Gold_Output_Layer'Range loop
                Output_Error (T) 
                  := new Real_Array (Actual_Output_Layer (T).all'Range);  
                
                Subtraction
                  (V_Out => Output_Error (T).all,
                   V1    => Actual_Output_Layer (T).all,
                   V2    => Gold_Output_Layer (T).all);
               
                for I in Gold_Output_Layer (T)'Range loop
                    Loss := Loss + (0.5 * ((Gold_Output_Layer (T) (I) - Actual_Output_Layer (T) (I)) ** 2));
                end loop;
                Loss := Loss / Real (Gold_Output_Layer (T)'Length);
                
            end loop;
            
        end return;
        
    end Calculate_Output_Error;
    
    procedure Weight_Update_SGD
      (Model      : in out Model_Type;
       Gradient   : in     Gradient_Type) is separate;
    
    procedure Weight_Update_ADAM 
      (Model      : in out Model_Type;
       Gradient   : in     Gradient_Type) is separate;
    
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
                
            when others => null;
        end case;
    end Weight_Update;
    
    procedure Backward
      (Model                     : in out Model_Type;
       CFN_Structure             : in out CFN_Structure_Type) is separate;
     
    procedure Initialize_CFN_Structure
      (CFN_Structure      : in out CFN_Structure_Type) is  
        
        Input_Layer_Size   : Positive renames CFN_Structure.Input_Layer_Size;
        Hidden_Layer_Size  : Positive renames CFN_Structure.Hidden_Layer_Size;
        Output_Layer_Size  : Positive renames CFN_Structure.Output_Layer_Size;
    begin
        
        CFN_Structure.Sequence_Hidden (CFN_Structure.First_Hidden_Sequence_Index) := new Real_Array
          (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);
            
        CFN_Structure.Sequence_Hidden_Derivative (CFN_Structure.First_Hidden_Sequence_Index) := new Real_Array ---useless
          (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);
        
        for T in CFN_Structure.First_Sequence_Index .. CFN_Structure.Last_Sequence_Index loop
            CFN_Structure.Sequence_Input (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Input_Layer_Size) -  1);
            
            CFN_Structure.Sequence_Input_Gradients (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Input_Layer_Size) -  1);
         
            CFN_Structure.Sequence_Forget_Activations (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);
         
            CFN_Structure.Sequence_Input_Activations (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);
         
            CFN_Structure.Sequence_Candidate_Activations (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);
            
            CFN_Structure.Sequence_Hidden_Activations (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);
         
            CFN_Structure.Sequence_Hidden (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);

            CFN_Structure.Sequence_Hidden_Derivative (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);

            CFN_Structure.Sequence_Output (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Output_Layer_Size) -  1);
                
            CFN_Structure.Sequence_Gold (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Output_Layer_Size) -  1);
            
            CFN_Structure.Sequence_Output_Error (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Output_Layer_Size) -  1);
        end loop;
          
    end Initialize_CFN_Structure;
    
    procedure Finalize_CFN_Structure
      (CFN_Structure : CFN_Structure_Type) is
    begin
        Free (CFN_Structure.Sequence_Input);
        Free (CFN_Structure.Sequence_Input_Gradients);
        Free (CFN_Structure.Sequence_Forget_Activations);
        Free (CFN_Structure.Sequence_Input_Activations);
        Free (CFN_Structure.Sequence_Candidate_Activations);
        Free (CFN_Structure.Sequence_Hidden_Activations);
        Free (CFN_Structure.Sequence_Hidden);
        Free (CFN_Structure.Sequence_Hidden_Derivative);
        Free (CFN_Structure.Sequence_Output);
        Free (CFN_Structure.Sequence_Gold);
        Free (CFN_Structure.Sequence_Output_Error);        
    end Finalize_CFN_Structure;
    
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
            raise CFN_Exception
              with "Loaded model is not initialized";
        end if;
    end Load;
    
end MLColl.Neural_Networks.Recurrent.CFN;
