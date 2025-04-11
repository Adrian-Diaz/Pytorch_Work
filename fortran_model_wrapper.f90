module fortran_model_wrapper
  use iso_c_binding, only: c_ptr, c_int
  implicit none
  interface
    function allocate_model_struct() bind(C, name="allocate_model_struct")
      import :: c_ptr
      type(c_ptr) allocate_model_struct
    end function allocate_model_struct
    subroutine delete_model_struct(ptr) bind(C, name="delete_model_struct")
      import :: c_ptr
      type(c_ptr), value :: ptr
    end subroutine delete_model_struct
    function setup_model(ptr) bind(C, name="setup_model")
      import :: c_ptr
      type(c_ptr), value :: ptr
    end function setup_model
    subroutine forward_propagate_model(data_ptr, batch_size, model_ptr) bind(C, name="forward_propagate_model")
      import :: c_ptr, c_int
      type(c_ptr), value :: data_ptr
      integer(c_int), value :: batch_size
      type(c_ptr), value :: model_ptr
    end subroutine forward_propagate_model
  end interface
end module fortran_model_wrapper