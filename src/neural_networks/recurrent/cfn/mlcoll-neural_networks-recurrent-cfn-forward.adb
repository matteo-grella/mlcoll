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

separate (MLColl.Neural_Networks.Recurrent.CFN)

procedure Forward
  (Model                   : in     Model_Type;
   CFN_Structure           : access CFN_Structure_Type) is

    Input_Layers              : Real_Array_Access_Array renames CFN_Structure.Sequence_Input;
    Forget_Activations        : Real_Array_Access_Array renames CFN_Structure.Sequence_Forget_Activations;
    Input_Activations         : Real_Array_Access_Array renames CFN_Structure.Sequence_Input_Activations;
    Candidate_Activations     : Real_Array_Access_Array renames CFN_Structure.Sequence_Candidate_Activations;
    Hidden_Activations        : Real_Array_Access_Array renames CFN_Structure.Sequence_Hidden_Activations;
    Hidden_Layers             : RNN_Array_Of_Float_Vectors_Type renames CFN_Structure.Sequence_Hidden;
    Output_Layers             : Real_Array_Access_Array renames CFN_Structure.Sequence_Output;
        
    
begin
    
    
    for T in Input_Layers'Range loop

        ---
        -- Forget Gate
        ---
       
        Addition (Forget_Activations (T).all, Model.Bf_In.all);
         
        for I in Hidden_Layers (T)'Range loop
            if Hidden_Layers (T - 1) (I) /= 0.0 then
                for J in Model.Wf_Rec'Range (2) loop
                    Forget_Activations (T) (J) := Forget_Activations (T) (J) 
                      + Hidden_Layers (T - 1) (I) * Model.Wf_Rec (I, J);
                end loop;
            end if;
        end loop;
         
        for I in Input_Layers (T)'Range loop
            if Input_Layers (T) (I) /= 0.0 then -- optimization
                for J in Model.Wf_In'Range (2) loop
                    Forget_Activations (T) (J) := Forget_Activations (T) (J) 
                      + Input_Layers (T) (I) * Model.Wf_In (I, J);
                end loop;
            end if;
        end loop;
         
        Activation
          (Sigmoid,
           Forget_Activations (T).all);
         
        ---
        -- Input Gate
        ---
         
        Addition (Input_Activations (T).all, Model.Bi_In.all);
          
        for I in Hidden_Layers (T)'Range loop
            if Hidden_Layers (T - 1) (I) /= 0.0 then
                for J in Model.Wi_Rec'Range (2) loop
                    Input_Activations (T) (J) := Input_Activations (T) (J) 
                      + Hidden_Layers (T - 1) (I) * Model.Wi_Rec (I, J);
                end loop;
            end if;
        end loop;
         
        for I in Input_Layers (T)'Range loop
            if Input_Layers (T) (I) /= 0.0 then -- optimization
                for J in Model.Wi_In'Range (2) loop
                    Input_Activations (T) (J) := Input_Activations (T) (J) 
                      + Input_Layers (T) (I) * Model.Wi_In (I, J);
                end loop;
            end if;
        end loop;
         
        -- Sigmoid Activation

        Activation (Sigmoid, Input_Activations (T).all);
         
        ---
        -- Candidate
        ---
         
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
            Hidden_Activations (T) (I) 
              := Hidden_Layers (T - 1) (I);
        end loop; -- todo more efficient
        
        Activation
          (Tanh,
           Hidden_Activations (T).all);
            
        for I in Hidden_Layers (T - 1)'Range loop
            Hidden_Layers (T) (I) 
              := Candidate_Activations (T) (I) * Input_Activations (T) (I) 
              + Hidden_Activations (T) (I) * Forget_Activations (T) (I) ;
        end loop;
        
        ---
        -- Output Layer
        ---
            
        VM_Product (Output_Layers (T).all, Hidden_Layers (T).all, Model.W_Out.all);

        Addition (Output_Layers (T).all, Model.B_Out.all);
        
    end loop;
    
end Forward;
    
