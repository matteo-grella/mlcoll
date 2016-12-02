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

with ARColl; use ARColl;
with ARColl.Numerics.Reals; use ARColl.Numerics.Reals;

package MLColl.Neural_Networks is
   
    type Activation_Function_Name_Type is
      (ReLU,
       Leaky_ReLU,
       Soft_Sign,
       Sigmoid,
       Tanh);

    type Learning_Rule_Type is
      (SGD,
       ADAM,
       ADAGRAD);

    type Features_Type_Type is
      (SPARSE,
       DENSE);

    type ADAM_Hyperparams_Type is record
        Beta1      : Real := 0.9;
        Beta2      : Real := 0.999;
        Beta1_Inv  : Real := 1.0 - 0.9;   -- 1.0 - Beta1;
        Beta2_Inv  : Real := 1.0 - 0.999; -- 1.0 - Beta2;
        Epsilon    : Real := 1.0E-8;
    end record;
    
    Default_ADAM_Hyperparams : constant ADAM_Hyperparams_Type 
      := (Beta1     => 0.9,
          Beta2     => 0.999,
          Beta1_Inv => 1.0 - 0.9,   -- 1.0 - Beta1;
          Beta2_Inv => 1.0 - 0.999, -- 1.0 - Beta2;
          Epsilon   => 1.0E-8);
    
    type Constraint_Hyperparams_Type is record
        Pi : Real     := 0.0;
        C  : Positive := 10;
    end record;
    
    Default_Constraint_Hyperparams : constant Constraint_Hyperparams_Type 
      := (Pi => 0.0, -- 0.1 standard parameter
          C  => 10);
    
    ----
    -- Normalization Functions
    ----

    function Softmax
      (V                : in  Real_Array;
       SoftMax_Vector   : out Real_Array) return Index_Type with Inline;

    procedure Softmax
      (V              : in  Real_Array;
       SoftMax_Vector : out Real_Array);
    
    ----
    -- Activation Functions
    ----

    procedure Soft_Sign
      (V : in out Real_Array) with Inline;

    procedure Sigmoid
      (V : in out Real_Array) with Inline;

    function Sigmoid
      (A : Real) return Real with Inline;
        
    procedure ReLU
      (V : in out Real_Array) with Inline;

    procedure Leaky_ReLU
      (V : in out Real_Array) with Inline;

    procedure Tanh
      (V : in out Real_Array) with Inline;

    ----
    -- Derivate
    ----

    procedure Soft_Sign_Deriv
      (Hidden_Layer       : in     Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) with Inline;

    procedure Sigmoid_Deriv
      (Hidden_Layer       : in     Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) with Inline;

    procedure ReLU_Deriv
      (Hidden_Layer       : in     Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) with Inline;

    procedure Leaky_ReLU_Deriv
      (Hidden_Layer       : in     Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) with Inline;

    procedure Activation
      (Activation_Function : in Activation_Function_Name_Type;
       V                   : in out Real_Array) with Inline;

    procedure Activation_Deriv
      (Activation_Function : in Activation_Function_Name_Type;
       V1                  : in     Real_Array;
       V2_Out              : in out Real_Array) with Inline;

    procedure Combined_Activation_Deriv
      (Activation_Function : in     Activation_Function_Name_Type;
       V1                  : in out Real_Array;
       V2                  : in out Real_Array) with Inline;

    procedure ReLU_Act_Deriv
      (Hidden_Layer       : in out Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) with Inline;

    procedure Tanh_Act_Deriv
      (Hidden_Layer       : in out Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) with Inline;

    procedure Sigmoid_Act_Deriv
      (Hidden_Layer       : in out Real_Array;
       Hidden_Layer_Deriv : in out Real_Array) with Inline;
        
end MLColl.Neural_Networks;

