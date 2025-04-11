program test
  use iso_c_binding, only: c_ptr, c_int
  use fortran_model_wrapper
  implicit none
  type(c_ptr) :: model_ptr
  integer(c_int) :: batch_size

  batch_size = 4
  ! Create a new C++ object
  model_ptr = allocate_model_struct()
  ! Set a value in the structure via the C++ wrapper
  call setup_model(model_ptr)
  print *, "Value of batch size is: ", batch_size
  ! forward propagate arbitrary data
  call forward_propagate_model(data_ptr, batch_size, model_ptr)
  ! Clean up the allocated object
  call delete_model_struct(model_ptr)
end program test