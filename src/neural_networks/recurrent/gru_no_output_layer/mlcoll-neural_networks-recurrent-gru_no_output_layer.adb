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

with ARColl.Numerics.Reals.Functions; use ARColl.Numerics.Reals.Functions;

package body MLColl.Neural_Networks.Recurrent.GRU_No_Output_Layer is
    
    package Stream_IO renames Ada.Streams.Stream_IO;
    
    procedure Initialize
      (Model                : in out Model_Type;
       Configuration        : in     Configuration_Type;
       Initialize_Weights   : in     Boolean := False) is
    begin
        
        if Model.Is_Initialized then
            raise GRU_Exception
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
            raise GRU_Exception
              with "Model not initialized";
        end if;
        
        Free (Model.Wc_In);
        Free (Model.Wr_In);
        Free (Model.Wz_In);
        Free (Model.Wc_Rec);
        Free (Model.Wr_Rec);
        Free (Model.Wz_Rec);
        Free (Model.Bc_In);
        Free (Model.Br_In);
        Free (Model.Bz_In);
        
        -- ADAM
        
        if Model.Configuration.Learning_Rule = ADAM then
            Free (Model.Wcm_In);
            Free (Model.Bcm_In);
            Free (Model.Wcm_Rec);        
            Free (Model.Wcv_In);
            Free (Model.Bcv_In);
            Free (Model.Wcv_Rec);
         
            Free (Model.Wzm_In);
            Free (Model.Bzm_In);
            Free (Model.Wzm_Rec);
            Free (Model.Wzv_In);
            Free (Model.Bzv_In);
            Free (Model.Wzv_Rec);
         
            Free (Model.Wrm_In);
            Free (Model.Brm_In); 
            Free (Model.Wrm_Rec);
            Free (Model.Wrv_In);
            Free (Model.Brv_In);
            Free (Model.Wrv_Rec);

            Model.Timestep := 0.0;
        end if;
        
        Model.Learning_Rate := Model.Configuration.Initial_Learning_Rate;
        
        Model.Is_Initialized := False;        
    end Finalize;
    
    procedure Initialize_Matrices
      (Model : in out Model_Type) is separate;
          
    procedure Forward
      (Model                   :     in Model_Type;
       --GRU_Structure           : in out GRU_Structure_Type) is
       GRU_Structure           : access GRU_Structure_Type) is separate;
    
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
                
            when ADAGRAD =>
                raise Constraint_Error with "ADAGRAD not implemented";
        end case;
    end Weight_Update;
    
    procedure Backward
      (Model                     : in out Model_Type;
       GRU_Structure             : in out GRU_Structure_Type) is separate;
     
    procedure Initialize_GRU_Structure
      (GRU_Structure      : in out GRU_Structure_Type) is  
        
       Input_Layer_Size   : Positive renames GRU_Structure.Input_Layer_Size;
       Hidden_Layer_Size  : Positive renames GRU_Structure.Hidden_Layer_Size;
    begin

        for T in GRU_Structure.First_Sequence_Index .. GRU_Structure.Last_Sequence_Index loop
            GRU_Structure.Sequence_Input (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Input_Layer_Size) -  1);
            
            GRU_Structure.Sequence_Input_Gradients (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Input_Layer_Size) -  1);
         
            GRU_Structure.Sequence_Reset_Activations (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);
         
            GRU_Structure.Sequence_Interpolate_Activations (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);
         
            GRU_Structure.Sequence_Candidate_Activations (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);
         
            GRU_Structure.Sequence_Hidden (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);

            GRU_Structure.Sequence_Hidden_Derivative (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);
            
            GRU_Structure.Sequence_Hidden_Error (T) := new Real_Array
              (Index_Type'First .. Index_Type'First + Index_Type (Hidden_Layer_Size) -  1);
        end loop;
          
    end Initialize_GRU_Structure;
    
    procedure Finalize_GRU_Structure
      (GRU_Structure : GRU_Structure_Type) is
    begin
        Free (GRU_Structure.Sequence_Input);
        Free (GRU_Structure.Sequence_Input_Gradients);
        Free (GRU_Structure.Sequence_Reset_Activations);
        Free (GRU_Structure.Sequence_Interpolate_Activations);
        Free (GRU_Structure.Sequence_Candidate_Activations);
        Free (GRU_Structure.Sequence_Hidden);
        Free (GRU_Structure.Sequence_Hidden_Derivative);
        Free (GRU_Structure.Sequence_Hidden_Error);
    end Finalize_GRU_Structure;
    
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
            raise GRU_Exception
              with "Loaded model is not initialized";
        end if;
    end Load;
    
end MLColl.Neural_Networks.Recurrent.GRU_No_Output_Layer;
