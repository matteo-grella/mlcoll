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

with ARColl.Numerics.Reals.Functions; use ARColl.Numerics.Reals.Functions;
--with Ada.Text_IO; use Ada.Text_IO;

package body MLColl.Neural_Networks is
    
    function Softmax
      (V                : in  Real_Array;
       SoftMax_Vector   : out Real_Array) return Index_Type is

        Max_Index : Index_Type := V'First;
        Sum       : Real := 0.0;
    begin
        
        -- Search max
        for I in V'First + 1 .. V'Last loop
            if V (I) > V (Max_Index) then
                Max_Index := I;
            end if;
        end loop;

        -- Exp & Sum
        for I in V'First .. V'Last loop
            SoftMax_Vector (I) := Exp (V (I) - V (Max_Index));
            Sum := Sum + SoftMax_Vector (I);
        end loop;

        -- Normalization
        for I in V'First .. V'Last loop
            SoftMax_Vector (I) := SoftMax_Vector (I) / Sum;
        end loop;

        return Max_Index;

    end Softmax;
   
    procedure Softmax
      (V              : in Real_Array;
       SoftMax_Vector : out Real_Array) is
       
        Arg_Max_Index : Extended_Index_Type := -1;
        pragma Unreferenced (Arg_Max_Index);
    begin
        Arg_Max_Index := Softmax
          (V              => V,
           SoftMax_Vector => SoftMax_Vector);
    end Softmax;
    
    ----
    -- Activation Functions
    ----
    
    procedure Soft_Sign
      (V : in out Real_Array) is
    begin
        for I in V'Range loop
            V (I) := V (I) / (1.0 + (abs V (I)));
        end loop;
    end Soft_Sign;

    procedure Sigmoid
      (V : in out Real_Array) is
    begin
        for I in V'Range loop
            V (I) := 1.0 / (1.0 + (Exp(- V(I))));
        end loop;
    end Sigmoid;
    
    function Sigmoid
      (A : Real) return Real is
    begin
        return 1.0 / (1.0 + (Exp(- A)));
    end Sigmoid;
    
    procedure ReLU
      (V : in out Real_Array) is
    begin
        for I in V'Range loop
            if V (I) <= 0.0 then 
                V(I) := 0.0;
            end if;
        end loop;
    end ReLU;
   
    procedure Leaky_ReLU
      (V : in out Real_Array) is
    begin
        for I in V'Range loop
            if V (I) <= 0.0 then 
                V(I) := 0.01 * V(I);
            end if;
        end loop;
    end Leaky_ReLU;

    procedure Tanh
      (V : in out Real_Array) is
    begin
        for I in V'Range loop
            V (I) := Tanh( V (I)) ;
        end loop;
    end Tanh;
    
    ----
    -- Derivate Activation Functions
    ----
    
    procedure Soft_Sign_Deriv
      (Hidden_Layer       : in Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) is
    begin
        for I in Hidden_Layer_Deriv'Range loop
            Hidden_Layer_Deriv (I) := (1.0 + (abs Hidden_Layer (I))) ** 2;
        end loop;
    end Soft_Sign_Deriv;
        
    procedure Sigmoid_Deriv
      (Hidden_Layer       : in Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) is
    begin
        for I in Hidden_Layer_Deriv'Range loop
            Hidden_Layer_Deriv (I) := Hidden_Layer (I) * (1.0 - Hidden_Layer (I));
        end loop;
    end Sigmoid_Deriv;
    
    procedure ReLU_Deriv
      (Hidden_Layer       : in Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) is
    begin
        for I in Hidden_Layer_Deriv'Range loop
            Hidden_Layer_Deriv (I) := (if Hidden_Layer (I) > 0.0 then 1.0 else 0.0); 
        end loop;
    end ReLU_Deriv;
    
    procedure Leaky_ReLU_Deriv
      (Hidden_Layer       : in Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) is
    begin
        for I in Hidden_Layer_Deriv'Range loop
            Hidden_Layer_Deriv (I) := (if Hidden_Layer (I) > 0.0 then 1.0 else 0.01); 
        end loop;
    end Leaky_ReLU_Deriv;
   
    procedure Tanh_Deriv
      (Hidden_Layer       : in Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) is
    begin
        for I in Hidden_Layer_Deriv'Range loop
            Hidden_Layer_Deriv (I) := 1.0 - ( Hidden_Layer (I) ** 2) ;
        end loop;
    end Tanh_Deriv;
    
    ----
    -- Combined Activations and Derivations
    ----
    
    procedure ReLU_Act_Deriv
      (Hidden_Layer       : in out Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) is
    begin
        for I in Hidden_Layer_Deriv'Range loop
            if Hidden_Layer (I) <= 0.0 then
                Hidden_Layer (I) := 0.0;
            end if;

            Hidden_Layer_Deriv (I) := (if Hidden_Layer (I) > 0.0 then 1.0 else 0.0);
        end loop;
    end ReLU_Act_Deriv;

    procedure Tanh_Act_Deriv
      (Hidden_Layer       : in out Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) is
    begin
        for I in Hidden_Layer_Deriv'Range loop
            Hidden_Layer (I) := Tanh( Hidden_Layer (I)) ;
            Hidden_Layer_Deriv (I) := 1.0 - ( Hidden_Layer (I) ** 2) ;
        end loop;
    end Tanh_Act_Deriv;

    procedure Sigmoid_Act_Deriv
      (Hidden_Layer       : in out Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) is
    begin
        for I in Hidden_Layer_Deriv'Range loop
            Hidden_Layer (I) := 1.0 / (1.0 + (Exp(- Hidden_Layer(I))));
            Hidden_Layer_Deriv (I) := Hidden_Layer (I) * (1.0 - Hidden_Layer (I));
        end loop;
    end Sigmoid_Act_Deriv;
    
    procedure Activation
      (Activation_Function : in Activation_Function_Name_Type;
       V                   : in out Real_Array) is
    begin
        case Activation_Function is
            when Soft_Sign  => Soft_Sign (V);    
            when ReLU       => ReLU (V);
            when Leaky_ReLU => Leaky_ReLU (V);
            when Tanh       => Tanh(V);
            when Sigmoid    => Sigmoid(V);
        end case;
    end Activation;
    
    procedure Activation_Deriv
      (Activation_Function : in Activation_Function_Name_Type;
       V1                  : in Real_Array;
       V2_Out              : in out Real_Array) is
    begin
        case Activation_Function is
            when Soft_Sign  => Soft_Sign_Deriv (V1, V2_Out);    
            when ReLU       => ReLU_Deriv (V1, V2_Out);
            when Leaky_ReLU => Leaky_ReLU_Deriv (V1, V2_Out);
            when Tanh       => Tanh_Deriv (V1, V2_Out);
            when Sigmoid    => Sigmoid_Deriv (V1, V2_Out);
        end case;
    end Activation_Deriv;
    
    procedure Combined_Activation_Deriv
      (Activation_Function : in Activation_Function_Name_Type;
       V1                  : in out Real_Array;
       V2                  : in out Real_Array) is
    begin
        case Activation_Function is
            when Soft_Sign   => Soft_Sign_Deriv (V1, V2);    
            when ReLU        => ReLU_Act_Deriv (V1, V2);
            when Tanh        => Tanh_Act_Deriv (V1, V2);
            when Sigmoid     => Sigmoid_Act_Deriv (V1, V2);
            when others => 
                raise Constraint_Error
                  with Activation_Function'Img & " Activation_Deriv not implemented";
        end case;
    end Combined_Activation_Deriv;
    
end MLColl.Neural_Networks;
