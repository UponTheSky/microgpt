use std::{
    cell::RefCell,
    collections::HashSet,
    iter::zip,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

use uuid::Uuid;

#[derive(Debug)]
struct Value {
    id: String,
    data: f64,
    grad: f64,
    children: Vec<Rc<RefCell<Value>>>,
    local_grads: Vec<f64>,
}

#[derive(Debug)]
pub struct ValueRef(Rc<RefCell<Value>>);

impl Clone for ValueRef {
    fn clone(&self) -> Self {
        ValueRef(Rc::clone(&self.0))
    }
}

impl ValueRef {
    pub fn new(data: f64, children: Vec<Rc<RefCell<Value>>>, local_grads: Vec<f64>) -> Self {
        ValueRef(Rc::new(RefCell::new(Value {
            id: Uuid::new_v4().to_string(),
            data,
            grad: 0.,
            children,
            local_grads,
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }
}

impl Add for ValueRef {
    type Output = ValueRef;

    fn add(self, rhs: Self) -> Self::Output {
        ValueRef::new(
            self.data() + rhs.data(),
            vec![Rc::clone(&self.0), Rc::clone(&rhs.0)],
            vec![1., 1.],
        )
    }
}

impl Mul for ValueRef {
    type Output = ValueRef;

    fn mul(self, rhs: Self) -> Self::Output {
        let self_data = self.data();
        let other_data = rhs.data();

        ValueRef::new(
            self_data * other_data,
            vec![Rc::clone(&self.0), Rc::clone(&rhs.0)],
            vec![other_data, self_data],
        )
    }
}

impl ValueRef {
    // no built-in __pow__ like Python
    pub fn pow(self, other: Self) -> Self {
        let self_data = self.data();
        let other_data = other.data();

        ValueRef::new(
            self_data.powf(other_data),
            vec![Rc::clone(&self.0)],
            vec![other_data * self_data.powf(other_data - 1.)],
        )
    }

    pub fn log(self) -> Self {
        let self_data = self.data();

        ValueRef::new(
            self_data.ln(),
            vec![Rc::clone(&self.0)],
            vec![1.0 / self_data],
        )
    }

    pub fn exp(self) -> Self {
        let self_data = self.data();

        ValueRef::new(
            self_data.exp(),
            vec![Rc::clone(&self.0)],
            vec![self_data.exp()],
        )
    }

    pub fn relu(self) -> Self {
        let self_data = self.data();

        ValueRef::new(
            if self_data > 0. { self_data } else { 0. },
            vec![Rc::clone(&self.0)],
            vec![if self_data > 0. { 1.0 } else { 0. }],
        )
    }

    pub fn backward(self) {
        let mut topo: Vec<ValueRef> = vec![];
        let mut visited = HashSet::new();

        self.0.borrow_mut().grad = 1.; // update earlier before build_topo borrows

        ValueRef::build_topo(self, &mut topo, &mut visited);
        topo.reverse();

        for v in topo {
            for (child, local_grad) in zip(
                v.0.borrow().children.iter(),
                v.0.borrow().local_grads.iter(),
            ) {
                child.borrow_mut().grad += local_grad * v.0.borrow().grad;
            }
        }
    }

    fn build_topo(value: ValueRef, topo: &mut Vec<ValueRef>, visited: &mut HashSet<String>) {
        let value_id = value.0.borrow().id.clone();

        if !visited.contains(&value_id) {
            visited.insert(value_id);

            for child in value.0.borrow().children.iter() {
                ValueRef::build_topo(ValueRef(Rc::clone(child)), topo, visited);
            }

            topo.push(value);
        }
    }
}

impl Neg for ValueRef {
    type Output = ValueRef;

    fn neg(self) -> Self::Output {
        self * ValueRef::new(-1., vec![], vec![])
    }
}

impl Sub for ValueRef {
    type Output = ValueRef;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Div for ValueRef {
    type Output = ValueRef;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(ValueRef::new(-1., vec![], vec![]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_computation() {
        // asset
        let a = ValueRef::new(2., vec![], vec![]);
        let b = ValueRef::new(3., vec![], vec![]);

        let c = a.clone() * b.clone();
        let L = c + a.clone();

        // action
        L.backward();

        // assert
        assert_eq!(a.0.borrow().grad, 4.0);
        assert_eq!(b.0.borrow().grad, 2.0)
    }
}
