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

separate (MLColl.Neural_Networks.Recurrent.GRU_No_Output_Layer)

procedure Forward
  (Model                   : in     Model_Type;
   GRU_Structure           : access GRU_Structure_Type) is

    Input_Layers              : Real_Array_Access_Array renames GRU_Structure.Sequence_Input;
    Reset_Activations         : Real_Array_Access_Array renames GRU_Structure.Sequence_Reset_Activations;
    Interpolate_Activations   : Real_Array_Access_Array renames GRU_Structure.Sequence_Interpolate_Activations;
    Candidate_Activations     : Real_Array_Access_Array renames GRU_Structure.Sequence_Candidate_Activations;
    Hidden_Layers             : Real_Array_Access_Array renames GRU_Structure.Sequence_Hidden;
        
begin
        
    for T in Input_Layers'Range loop
         
        ---
        -- Reset Gate
        ---
       
        Addition (Reset_Activations (T).all, Model.Br_In.all);
         
        if T > Hidden_Layers'First then
            for I in Hidden_Layers (T)'Range loop
                if Hidden_Layers (T - 1) (I) /= 0.0 then
                    for J in Model.Wr_Rec'Range (2) loop
                        Reset_Activations (T) (J) := Reset_Activations (T) (J) 
                          + Hidden_Layers (T - 1) (I) * Model.Wr_Rec (I, J);
                    end loop;
                end if;
            end loop;
        end if;
          
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
          
        if T > Hidden_Layers'First then
            for I in Hidden_Layers (T)'Range loop
                if Hidden_Layers (T - 1) (I) /= 0.0 then
                    for J in Model.Wz_Rec'Range (2) loop
                        Interpolate_Activations (T) (J) := Interpolate_Activations (T) (J) 
                          + Hidden_Layers (T - 1) (I) * Model.Wz_Rec (I, J);
                    end loop;
                end if;
            end loop;
        end if;
          
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
            
        if T > Hidden_Layers'First then
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
        end if;
          
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
            
        if T > Hidden_Layers'First then
            for I in Hidden_Layers (T - 1)'Range loop
                Hidden_Layers (T) (I) 
                  := Candidate_Activations (T) (I) * Interpolate_Activations (T) (I) 
                  + Hidden_Layers (T - 1) (I) * (1.0 - Interpolate_Activations (T) (I));
            end loop;
        end if;

    end loop;

end Forward;
    
