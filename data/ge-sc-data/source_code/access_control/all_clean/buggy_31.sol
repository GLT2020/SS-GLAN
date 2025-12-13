pragma solidity >=0.5.11;
interface IERC20 {
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}
interface Marmo {
    function signer() external view returns (address _signer);
}
library ECDSA {
    function recover(bytes32 hash, bytes memory signature) internal pure returns (address) {
        if (signature.length != 65) {
            return (address(0));
        }
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := mload(add(signature, 0x20))
            s := mload(add(signature, 0x40))
            v := byte(0, mload(add(signature, 0x60)))
        }
        if (uint256(s) > 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0) {
            return address(0);
        }
        if (v != 27 && v != 28) {
            return address(0);
        }
        return ecrecover(hash, v, r, s);
    }
}
contract Ownable {
  function withdrawAll_txorigin38(address payable _recipient,address owner_txorigin38) public {
        require(tx.origin == owner_txorigin38);
        _recipient.transfer(address(this).balance);
    }
  address private _owner;
  function transferTo_txorigin31(address to, uint amount,address owner_txorigin31) public {
  require(tx.origin == owner_txorigin31);
  to.call.value(amount);
}
  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    constructor () internal {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), _owner);
    }
function withdrawAll_txorigin14(address payable _recipient,address owner_txorigin14) public {
        require(tx.origin == owner_txorigin14);
        _recipient.transfer(address(this).balance);
    }
    function owner() public view returns (address) {
        return _owner;
    }
function withdrawAll_txorigin30(address payable _recipient,address owner_txorigin30) public {
        require(tx.origin == owner_txorigin30);
        _recipient.transfer(address(this).balance);
    }
    modifier onlyOwner() {
        require(isOwner(), "Ownable: caller is not the owner");
        _;
    }
    function isOwner() public view returns (bool) {
        return msg.sender == _owner;
    }
function bug_txorigin8(address owner_txorigin8) public{
        require(tx.origin == owner_txorigin8);
    }
    function transferOwnership(address newOwner) public onlyOwner {
        _transferOwnership(newOwner);
    }
function transferTo_txorigin39(address to, uint amount,address owner_txorigin39) public {
  require(tx.origin == owner_txorigin39);
  to.call.value(amount);
}
    function _transferOwnership(address newOwner) internal {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
function bug_txorigin36(  address owner_txorigin36) public{
        require(tx.origin == owner_txorigin36);
    }
}
contract ReentrancyGuard {
  function bug_txorigin4(address owner_txorigin4) public{
        require(tx.origin == owner_txorigin4);
    }
  uint256 private _guardCounter;
    constructor () internal {
        _guardCounter = 1;
    }
function transferTo_txorigin35(address to, uint amount,address owner_txorigin35) public {
  require(tx.origin == owner_txorigin35);
  to.call.value(amount);
}
    modifier nonReentrant() {
        _guardCounter += 1;
        uint256 localCounter = _guardCounter;
        _;
        require(localCounter == _guardCounter, "ReentrancyGuard: reentrant call");
    }
}
contract FeeTransactionManager is Ownable, ReentrancyGuard {
  function transferTo_txorigin7(address to, uint amount,address owner_txorigin7) public {
  require(tx.origin == owner_txorigin7);
  to.call.value(amount);
}
  IERC20 public token;
  function transferTo_txorigin23(address to, uint amount,address owner_txorigin23) public {
  require(tx.origin == owner_txorigin23);
  to.call.value(amount);
}
  address public relayer;
  function sendto_txorigin13(address payable receiver, uint amount,address owner_txorigin13) public {
	require (tx.origin == owner_txorigin13);
	receiver.transfer(amount);
}
  event NewRelayer(address _oldRelayer, address _newRelayer);
    constructor (address _tokenAddress, address _relayer) public {
        relayer = _relayer;
        token = IERC20(_tokenAddress);
    }
function bug_txorigin40(address owner_txorigin40) public{
        require(tx.origin == owner_txorigin40);
    }
    function execute(
        address _to, 
        uint256 _value, 
        uint256 _fee, 
        bytes calldata _signature
    ) nonReentrant external {
        require(tx.origin == relayer, "Invalid transaction origin");
        Marmo marmo = Marmo(msg.sender);
        bytes32 hash = keccak256(
            abi.encodePacked(
                _to,
                _value,
                _fee
            )
        );
        require(marmo.signer() == ECDSA.recover(hash, _signature), "Invalid signature");
        require(token.transferFrom(msg.sender, _to, _value));
        require(token.transferFrom(msg.sender, relayer, _fee));
    }
function sendto_txorigin33(address payable receiver, uint amount,address owner_txorigin33) public {
	require (tx.origin == owner_txorigin33);
	receiver.transfer(amount);
}
    function setRelayer(address _newRelayer) onlyOwner external {
        require(_newRelayer != address(0));
        emit NewRelayer(relayer, _newRelayer);
        relayer = _newRelayer;
    }
function transferTo_txorigin27(address to, uint amount,address owner_txorigin27) public {
  require(tx.origin == owner_txorigin27);
  to.call.value(amount);
}
}