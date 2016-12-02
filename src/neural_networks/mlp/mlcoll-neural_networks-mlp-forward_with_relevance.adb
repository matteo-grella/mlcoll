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

separate (MLColl.Neural_Networks.MLP)

procedure Forward_With_Relevance
  (Model         : in Model_Type;
   NN_Structure  : in out NN_Structure_Type) is
          
    NN : NN_Structure_Type renames NN_Structure;
        
    -- Contributes

    Input_Contributes : Real_Matrix
      (Index_Type'First .. NN.Input_Layer_Last, 
       Index_Type'First .. NN.Hidden_Layer_Last)
      := (others => (others => 0.0));
                          
    Hidden_Contributes : Real_Matrix
      (Index_Type'First .. NN.Hidden_Layer_Last, 
       Index_Type'First .. NN.Output_Layer_Last)
      := (others => (others => 0.0));

    procedure Calculate_Input_Relevance is

        Hidden_Relevance        : Real_Array
          (NN.Hidden_Layer'Range) := (others => 0.0);
        
        Output_Relevance        : constant Real_Array
          (NN.Output_Layer'Range) := (others => 1.0 / Real (NN.Output_Layer_Size));

        procedure Propagate_Relevance
          (Next_Layer_Relevance  : in     Real_Array;
           Contributes           : in out Real_Matrix;
           Cur_Layer_Relevance   : in out Real_Array) with Inline is

        begin
            
            ---
            -- Propagate Next Relevance to Current Contributes
            ---
                
            for J in Contributes'Range (2) loop
                for I in Contributes'Range (1) loop
                    Contributes (I, J) := Contributes (I, J) * Next_Layer_Relevance (J);
                end loop;
            end loop;

            ---
            -- Set Current Layer Relevance
            ---
                
            for I in Contributes'Range (1) loop
                declare
                    Contribute_I_Sum : Real := 0.0;
                begin
                        
                    for J in Contributes'Range (2) loop
                        Contribute_I_Sum := Contribute_I_Sum + Contributes (I, J);
                    end loop;
                        
                    Cur_Layer_Relevance (I) := Contribute_I_Sum;
                end;
            end loop;
                
        end Propagate_Relevance;
                
    begin

        -- Propagate Relevance from Output Layer to Hidden Layer
            
        Propagate_Relevance
          (Contributes          => Hidden_Contributes,
           Next_Layer_Relevance => Output_Relevance,
           Cur_Layer_Relevance  => Hidden_Relevance);
            
        -- Propagate Relevance from Hidden Layer to Input Layer
            
        Propagate_Relevance
          (Contributes          => Input_Contributes,
           Next_Layer_Relevance => Hidden_Relevance,
           Cur_Layer_Relevance  => NN.Input_Relevance);

    end Calculate_Input_Relevance;
           
    procedure Calculate_Contributes
      (Weight            : in     Real_Matrix;
       Bias              : in     Real_Array;           
       Cur_Layer_Size    : in     Positive;
       Cur_Layer         : in     Real_Array;
       Next_Layer        :    out Real_Array;
       Contributes       :    out Real_Matrix) is
            
        Eps : constant Real := 0.01 / Real (Cur_Layer_Size);
        
    begin
            
        for J in Next_Layer'Range loop

            declare
                B_J : constant Real := Bias (J) / Real (Cur_Layer_Size);
                        
                Contribute_J_Sum : Real := 0.0;

                Contribute_J_Norm     : Real_Array (Cur_Layer'Range) := (others => 0.0);
                Contribute_J_Norm_Sum : Real := 0.0;
            begin
                        
                -- Dot Product
                        
                for I in Cur_Layer'Range loop
                        
                    if Cur_Layer (I) /= 0.0 then
                        Contributes (I, J) := Cur_Layer (I) * Weight (I, J) + B_J;
                    else
                        Contributes (I, J) := B_J;
                    end if;
                                
                    Contribute_J_Sum := Contribute_J_Sum + Contributes (I, J);
                        
                end loop;
                    
                -- Normalize and Sum Contributes
                        
                for I in Cur_Layer'Range  loop            
                    Contribute_J_Norm (I) 
                      := Contributes (I, J) 
                      + (if Contribute_J_Sum >= 0.0 then Eps else -Eps);
                      
                    Contribute_J_Norm_Sum := Contribute_J_Norm_Sum + Contribute_J_Norm (I);
                end loop;
                
                -- Set Contributes
                        
                for I in Cur_Layer'Range loop
                    Contributes (I, J)  := Contribute_J_Norm (I) / Contribute_J_Norm_Sum;
                end loop;
                
                -- Set Next Layer
                        
                Next_Layer (J) := Contribute_J_Sum;
                        
            end;
                    
        end loop;
            
    end Calculate_Contributes;
        
begin

    ---
    -- Calculate Contributes between Input Layer and Hidden Layer
    ---
        
    Calculate_Contributes 
      (Weight         => Model.W_In.all,
       Bias           => Model.B_In.all,
       Cur_Layer_Size => NN_Structure.Input_Layer_Size,
       Cur_Layer      => NN_Structure.Input_Layer,
       Next_Layer     => NN_Structure.Hidden_Layer,
       Contributes    => Input_Contributes);

    ---
    -- Hidden Layer Activation
    ---
        
    NN.Activated_Hidden_Layer := NN.Hidden_Layer;
          
    Activation 
      (Model.Configuration.Activation_Function_Name, 
       NN.Activated_Hidden_Layer);

    ---
    -- Calculate contributes between Hidden Layer and Output Layer
    ---
  
    Calculate_Contributes 
      (Weight         => Model.W_Out.all,
       Bias           => Model.B_Out.all,
       Cur_Layer_Size => NN_Structure.Hidden_Layer_Size,
       Cur_Layer      => NN_Structure.Activated_Hidden_Layer,
       Next_Layer     => NN_Structure.Output_Layer,
       Contributes    => Hidden_Contributes);
        
    ---
    -- Calculate Relevance
    ---
          
    Calculate_Input_Relevance;
        
end Forward_With_Relevance;
