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

with Ada.Finalization;
with Ada.Containers.Vectors;

package MLColl.Neural_Networks.Datasets is
    
    type Example_Type is new Ada.Finalization.Controlled with
        record
            Features      : Real_Array_Access;
            Outcome_Index : Index_Type;
        end record;
    
    procedure Initialize (Example : in out Example_Type);
    procedure Adjust     (Example : in out Example_Type);
    procedure Finalize   (Example : in out Example_Type);
    
    package Example_Vectors is new
      Ada.Containers.Vectors
        (Index_Type   => Index_Type,
         Element_Type => Example_Type);
    
    subtype Dataset_Type is Example_Vectors.Vector;
    
end MLColl.Neural_Networks.Datasets;
