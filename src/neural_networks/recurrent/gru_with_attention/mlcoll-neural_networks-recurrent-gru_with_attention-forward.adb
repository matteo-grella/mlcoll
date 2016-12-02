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

separate (MLColl.Neural_Networks.Recurrent.GRU_With_Attention)

procedure Forward
  (Model                   : in     Model_Type;
   GRU_Structure           : access GRU_Structure_Type) is

    Input_Layers              : Real_Array_Access_Array renames GRU_Structure.Sequence_Input;
    Reset_Activations         : Real_Array_Access_Array renames GRU_Structure.Sequence_Reset_Activations;
    Interpolate_Activations   : Real_Array_Access_Array renames GRU_Structure.Sequence_Interpolate_Activations;
    Candidate_Activations     : Real_Array_Access_Array renames GRU_Structure.Sequence_Candidate_Activations;
    Hidden_Layers             : RNN_Array_Of_Float_Vectors_Type renames GRU_Structure.Sequence_Hidden;
    Output_Layers             : Real_Array_Access_Array renames GRU_Structure.Sequence_Output;
        
    -- Temporary sequences for Attention Layer
    Temp_A                    : Real_Array_Access_Array (Input_Layers'First ..  Input_Layers'Last) := (others => null);
    Temp_S                    : Real_Array_Access_Array (Input_Layers'First ..  Input_Layers'Last) := (others => null);
    Temp_Softmax              : Real_Array_Access_Array (Input_Layers'First ..  Input_Layers'Last) := (others => null);
    
begin

    -- Precalculate Attention Layers 
    
    for T in Input_Layers'Range loop
        
        Temp_A(T) := new Real_Array
          (Index_Type'First .. Index_Type'First + Index_Type (GRU_Structure.Hidden_Layer_Size) -  1);
    
        Temp_S (T) := new Real_Array
          (Index_Type'First .. Index_Type'First + Index_Type (GRU_Structure.Hidden_Layer_Size) -  1);
        
        VM_Product
          (V_Out => Temp_A(T).all,
           V_In  => Input_Layers(T).all , 
           M_In  => Model.Wa_In.all);
        
        Addition (Temp_A(T).all, Model.Ba_In.all);
        
        VM_Product
          (V_Out => GRU_Structure.Sequence_Attention_Hidden_Layers_B(T).all,
           V_In  => Input_Layers(T).all , 
           M_In  => Model.Wb_In.all);
        
        Activation (Tanh, GRU_Structure.Sequence_Attention_Hidden_Layers_B (T).all);
        
        Temp_Softmax (T) := new Real_Array
          (GRU_Structure.First_Sequence_Index .. GRU_Structure.First_Sequence_Index + T);
    end loop;
    
        
    for T in Input_Layers'Range loop
        
        --Put_Line(T'Img);
        
        ---
        -- Reset Gate
        ---
       
        Addition (Reset_Activations (T).all, Model.Br_In.all);
         
        for I in Hidden_Layers (T)'Range loop
            if Hidden_Layers (T - 1) (I) /= 0.0 then
                for J in Model.Wr_Rec'Range (2) loop
                    Reset_Activations (T) (J) := Reset_Activations (T) (J) 
                      + Hidden_Layers (T - 1) (I) * Model.Wr_Rec (I, J);
                end loop;
            end if;
        end loop;
         
        for I in Input_Layers (T)'Range loop
            if Input_Layers (T) (I) /= 0.0 then -- optimization
                for J in Model.Wr_In'Range (2) loop
                    Reset_Activations (T) (J) := Reset_Activations (T) (J) 
                      + Input_Layers (T) (I) * Model.Wr_In (I, J);
                end loop;
            end if;
        end loop;
         
        Activation
          (Sigmoid,
           Reset_Activations (T).all);
         
        ---
        -- Interpolate Gate
        ---
         
        Addition (Interpolate_Activations (T).all, Model.Bz_In.all);
          
        for I in Hidden_Layers (T)'Range loop
            if Hidden_Layers (T - 1) (I) /= 0.0 then
                for J in Model.Wz_Rec'Range (2) loop
                    Interpolate_Activations (T) (J) := Interpolate_Activations (T) (J) 
                      + Hidden_Layers (T - 1) (I) * Model.Wz_Rec (I, J);
                end loop;
            end if;
        end loop;
         
        for I in Input_Layers (T)'Range loop
            if Input_Layers (T) (I) /= 0.0 then -- optimization
                for J in Model.Wz_In'Range (2) loop
                    Interpolate_Activations (T) (J) := Interpolate_Activations (T) (J) 
                      + Input_Layers (T) (I) * Model.Wz_In (I, J);
                end loop;
            end if;
        end loop;
         
        -- Sigmoid activation
        Activation (Sigmoid, Interpolate_Activations (T).all);
         
        ---
        -- Candidate
        ---
         
        Addition (Candidate_Activations (T).all, Model.Bc_In.all);
            
        declare
            Reset_Hidden_Product : Real_Array 
              (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) -  1);
        begin
            for I in Hidden_Layers (T - 1)'Range loop
                if Hidden_Layers (T - 1) (I) /= 0.0 then
                    Reset_Hidden_Product (I) := Reset_Activations (T) (I) * Hidden_Layers (T - 1) (I);
                end if;
            end loop;
         
            for I in Hidden_Layers (T)'Range loop
                if Hidden_Layers (T - 1) (I) /= 0.0 then
                    for J in Model.Wc_Rec'Range (2) loop
                        Candidate_Activations (T) (J) := Candidate_Activations (T) (J) + Reset_Hidden_Product (I) * Model.Wc_Rec (I, J);
                    end loop;
                end if;
            end loop;   
        end;
         
        for I in Input_Layers (T)'Range loop
            if Input_Layers (T) (I) /= 0.0 then -- optimization
                for J in Model.Wc_In'Range (2) loop
                    Candidate_Activations (T) (J) := Candidate_Activations (T) (J) + Input_Layers (T) (I) * Model.Wc_In (I, J);
                end loop;
            end if;
        end loop;
         
        -- Tanh activation
        Activation (Tanh, Candidate_Activations (T).all);
         
        ---
        -- Current Hidden Layer
        ---
            
        for I in Hidden_Layers (T - 1)'Range loop
            Hidden_Layers (T) (I) 
              := Candidate_Activations (T) (I) * Interpolate_Activations (T) (I) 
              + Hidden_Layers (T - 1) (I) * (1.0 - Interpolate_Activations (T) (I));
        end loop;
        
        VM_Product
          (V_Out => Temp_S(T).all,
           V_In  => Hidden_Layers (T - 1).all, 
           M_In  => Model.Wa_Rec.all);
        
        --- 
        -- Attention Layer
        ---
        
        for I in GRU_Structure.First_Sequence_Index .. GRU_Structure.First_Sequence_Index + T loop
            declare
                Attention_Score : Real := 0.0; 
            begin
                --Put_Line(T'Img & "  I : " & " " & I'Img );
                for J in Temp_S (I)'First .. Temp_S (I)'Last loop
                    
                    GRU_Structure.Sequence_Attention_Hidden_Layers_A (T) (I) (J) := Temp_S (I) (J) + Temp_A (I) (J);
                    --Put (" @" & J'Img & " " & Temp_A (I) (J)'Img);
                    
                    --Put (" #" & J'Img & " " & GRU_Structure.Sequence_Attention_Hidden_Layers_A (T) (I) (J)'Img);
                    
                end loop;
                
                --New_Line;
                --New_Line;
                
                Activation (Tanh, GRU_Structure.Sequence_Attention_Hidden_Layers_A (T) (I).all);
                
                for J in Model.A_In'Range loop
                    Attention_Score := Attention_Score + 
                      Model.A_In (J) * GRU_Structure.Sequence_Attention_Hidden_Layers_A (T) (I) (J);
                end loop;
                
                
                Temp_Softmax (T) (I) := Attention_Score;
                --for J in Temp_Softmax(T)'Range loop
                --    Put_Line(J'Img & " " & Attention_Score'Img);
                --end loop;
            end;
        end loop;
        
        Softmax
          (V              => Temp_Softmax (T).all,
           SoftMax_Vector => GRU_Structure.Sequence_Attention_Scores (T).all);
        
        -- Debug
        -- for I in GRU_Structure.Sequence_Attention_Scores (T)'Range loop
        --     Put (GRU_Structure.Sequence_Attention_Scores (T) (I)'Img & " " );            
        -- end loop;
        -- New_Line;       
        
        for I in GRU_Structure.Sequence_Attention_Scores (T)'Range loop
            declare
                Temp_B : Real_Array_Access := new Real_Array
                  (Index_Type'First .. Index_Type'First + Index_Type (Model.Configuration.Hidden_Layer_Size) -  1);    
            begin
                for J in  Temp_B'Range loop
                    Temp_B (J) := GRU_Structure.Sequence_Attention_Hidden_Layers_B (T) (J) * 
                      GRU_Structure.Sequence_Attention_Scores (T) (I);
                    
                    --Put (" B" & J'Img & " " & Temp_B (J)'Img);    
                    --Put (" " & J'Img & " " & GRU_Structure.Sequence_Attention_Hidden_Layers_B (T) (J)'Img);
                end loop;
                
                Addition (GRU_Structure.Sequence_Attention_Layers (T).all, Temp_B.all);
                
                --New_Line;
                --New_Line;
             
                Free (Temp_B);
            end;
            
        end loop;
        
        for I in Hidden_Layers (T)'Range loop
            Hidden_Layers (T) (I) :=  Hidden_Layers (T) (I) + GRU_Structure.Sequence_Attention_Layers (T) (I); --  --   
        end loop;
        
        ---
        -- Output Layer
        ---
            
        VM_Product (Output_Layers (T).all, Hidden_Layers (T).all, Model.W_Out.all);

        Addition (Output_Layers (T).all, Model.B_Out.all);
        
    end loop;
        
    Free (Temp_A);
    Free (Temp_S);
    Free (Temp_Softmax);

end Forward;

