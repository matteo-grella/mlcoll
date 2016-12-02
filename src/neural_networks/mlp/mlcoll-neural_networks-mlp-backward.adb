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

procedure Backward
  (Model                : in out Model_Type;
   NN_Structure         : in out NN_Structure_Type;
   Gradient             : in out Gradient_Type;
   Accumulate_Gradients : in Boolean := False) is
    
    Hidden_Layer_Deriv : Real_Array (NN_Structure.Hidden_Layer'Range);

    Local_GB_In : Real_Array (Gradient.B_In'Range) := (others => 0.0);
begin
        
    Gradient.Count := (if Accumulate_Gradients then Gradient.Count + 1 else 1);
    
    -- GB_Out
    if Accumulate_Gradients then
        Addition (V1_Out => Gradient.B_Out,
                  V2     => NN_Structure.Output_Error);
    else
        Gradient.B_Out := NN_Structure.Output_Error;
    end if;
    
    -- GW_Out   
    if Accumulate_Gradients then
        VV_Add_Product
          (M_Out => Gradient.W_Out,
           V1    => NN_Structure.Activated_Hidden_Layer,
           V2    => NN_Structure.Output_Error);
    else
        VV_Product
          (M_Out => Gradient.W_Out,
           V1    => NN_Structure.Activated_Hidden_Layer,
           V2    => NN_Structure.Output_Error);
    end if;
    
    -- Hidden_Layer_Deriv
    Activation_Deriv
      (Activation_Function => Model.Configuration.Activation_Function_Name,
       V1                  => NN_Structure.Activated_Hidden_Layer,
       V2_Out              => Hidden_Layer_Deriv);
        
    -- GB_In
    MV_Product
      (V_Out => Local_GB_In,
       M_In  => Model.W_Out.all,
       V_In  => NN_Structure.Output_Error);
        
    case Model.Configuration.Activation_Function_Name is
        when Soft_Sign => 
            Element_Division 
              (V1_Out => Local_GB_In,
               V2     => Hidden_Layer_Deriv);

        when others => 
            Element_Product  
              (V1_Out => Local_GB_In,
               V2     => Hidden_Layer_Deriv);
    end case;
        
    -- GW_In
    if Accumulate_Gradients then
        VV_Add_Product
          (M_Out => Gradient.W_In,
           V1    => NN_Structure.Input_Layer,
           V2    => Local_GB_In);
    else
        VV_Product
          (M_Out => Gradient.W_In,
           V1    => NN_Structure.Input_Layer,
           V2    => Local_GB_In);
    end if;
    
    if Accumulate_Gradients then
        Addition 
          (V1_Out => Gradient.B_In,
           V2     => Local_GB_In);
    else
        Gradient.B_In := Local_GB_In;
    end if;
    
    -- GB_I
    NN_Structure.Input_Error := (others => 0.0);
    
    for I in NN_Structure.Input_Layer'Range loop
        if NN_Structure.Input_Layer (I) /= 0.0 then -- optimization    
            Gradient.Input_Non_Zero (I) := True;
              
            for J in Local_GB_In'Range loop
                if Local_GB_In (J) /= 0.0 then
                    NN_Structure.Input_Error (I) := NN_Structure.Input_Error (I) + Model.W_In (I, J) * Local_GB_In (J);
                end if;
            end loop;
        end if;        
    end loop;
        
end Backward;
