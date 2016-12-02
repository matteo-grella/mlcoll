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

procedure Forward
  (Model         : in Model_Type;
   NN_Structure  : access NN_Structure_Type) is
--   NN_Structure  : in out NN_Structure_Type) is
begin
        
    if Model.Configuration.Features_Type = SPARSE then
        for I of NN_Structure.Sparse_Input loop
            Addition
              (V_Out => NN_Structure.Hidden_Layer,
               M     => Model.W_In.all,
               M_Row => I);
        end loop;
    else
        VM_Product
          (V_Out => NN_Structure.Hidden_Layer,
           V_In  => NN_Structure.Input_Layer, 
           M_In  => Model.W_In.all);
    end if;
    
    Addition 
      (V1_Out => NN_Structure.Hidden_Layer, 
       V2     => Model.B_In.all);
    
    NN_Structure.Activated_Hidden_Layer := NN_Structure.Hidden_Layer;
          
    Activation 
      (Model.Configuration.Activation_Function_Name, 
       NN_Structure.Activated_Hidden_Layer);
        
    VM_Product
      (V_Out => NN_Structure.Output_Layer, 
       V_In  => NN_Structure.Activated_Hidden_Layer, 
       M_In  => Model.W_Out.all);

    Addition 
      (V1_Out => NN_Structure.Output_Layer, 
       V2     => Model.B_Out.all);
        
end Forward;
