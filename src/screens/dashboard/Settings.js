import React, { useState } from 'react';
import { Card, Table, Button, Modal, Form } from 'react-bootstrap'

const Settings = () => {
    const [show, setShow] = useState(false);

    const handleShow = () => setShow(true);
    const handleClose = () => setShow(false);

    return (
        <div style={{ display: 'flex', justifyContent: 'center' }}>
            <Modal show={show} onHide={handleClose}>
                <Modal.Header closeButton>
                <Modal.Title>Add New Device</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    <Form>
                        <Form.Group className="mb-3" controlId="formBasicEmail">
                            <Form.Label>Device Name</Form.Label>
                            <Form.Control type="text" placeholder="Jack's Android" />
                        </Form.Group>

                        <Form.Group className="mb-3" controlId="formBasicPassword">
                            <Form.Label>Address</Form.Label>
                            <Form.Control type="text" placeholder="127.0.0.1:7829" />
                        </Form.Group>
                    </Form>
                </Modal.Body>
                <Modal.Footer>
                <Button variant="secondary" onClick={handleClose}>
                    Close
                </Button>
                <Button variant="primary" onClick={handleClose}>
                    Save
                </Button>
                </Modal.Footer>
            </Modal>
            <div style={{ marginTop: '25px', width: '100%'}}>
                <Card>
                    <Card.Body>
                        <div style={{fontSize: '24px', fontWeight: 'bold'}} className="devices-title">
                            Devices
                        </div>
                        <div style={{fontSize: '18x', marginTop: '12px'}} className="devices-list">
                            <Table striped bordered hover style={{fontSize: '18x'}}>
                                <thead>
                                    <tr>
                                    <th>#</th>
                                    <th>Device Name</th>
                                    <th>Activity</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                    <td>1</td>
                                    <td>Device_1</td>
                                    <td>ONLINE</td>
                                    </tr>
                                </tbody>
                            </Table>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'right'}} className="devices-addButton">
                            <Button variant="primary" onClick={handleShow}>Add</Button>
                        </div>
                    </Card.Body>
                </Card>
            </div>
        </div>
    );
}

export default Settings;
